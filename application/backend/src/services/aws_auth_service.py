"""Discover and validate backend-local AWS credentials via polymorphic providers."""

import asyncio
import shlex
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import httpx
import yaml
from botocore.exceptions import BotoCoreError

from core.security import require_aws_feature_active
from schemas.aws_auth import (
    AwsAuthCapabilities,
    AwsAuthenticationIssue,
    AwsCredentialSelection,
    AwsIdentity,
    is_robot_enrolled,
)
from schemas.operational_issue import OperationalIssue
from schemas.remote_trainer import AwsBatchConnection, CloudProviderError
from settings import get_settings


class IotCertificateError(BotoCoreError):
    """Raised when robot X.509 certificate configuration or credential exchange fails."""

    fmt = "{detail}"

    def __init__(self, *, reason: str, title: str, detail: str) -> None:
        self.reason = reason
        self.title = title
        self.detail = detail
        super().__init__(detail=detail)


@dataclass(frozen=True)
class GreengrassDeviceConfig:
    """Parsed robot identity and endpoint settings from Greengrass config.yaml."""

    certificate_file_path: Path
    private_key_path: Path
    root_ca_path: Path
    thing_name: str
    iot_cred_endpoint: str
    iot_role_alias: str | None = None
    aws_region: str | None = None


def credential_error_reason(error: Exception) -> str | None:
    """Distinguish authentication failures from network and service failures."""
    from botocore.exceptions import (
        ClientError,
        LoginInsufficientPermissions,
        LoginRefreshRequired,
        LoginTokenLoadError,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
        SSOTokenLoadError,
        TokenRetrievalError,
        UnauthorizedSSOTokenError,
    )

    if isinstance(error, IotCertificateError):
        return error.reason
    if isinstance(error, ProfileNotFound):
        return "aws_profile_not_found"
    if isinstance(error, (NoCredentialsError, PartialCredentialsError)):
        return "aws_credentials_missing"
    code = error.response.get("Error", {}).get("Code") if isinstance(error, ClientError) else None
    if isinstance(
        error,
        (SSOTokenLoadError, UnauthorizedSSOTokenError, TokenRetrievalError, LoginRefreshRequired, LoginTokenLoadError),
    ) or code in {
        "ExpiredToken",
        "ExpiredTokenException",
        "RequestExpired",
        "InvalidClientTokenId",
        "InvalidIdentityToken",
    }:
        return "aws_credentials_expired"
    if isinstance(error, LoginInsufficientPermissions) or code in {
        "AccessDenied",
        "AccessDeniedException",
        "UnauthorizedOperation",
    }:
        return "aws_access_denied"
    return None


def aws_auth_error_detail(error: Exception) -> str:
    """Return actionable credential errors without disclosing SDK exception details."""
    from botocore.exceptions import SSOTokenLoadError, UnauthorizedSSOTokenError

    if isinstance(error, IotCertificateError):
        return error.detail
    reason = credential_error_reason(error)
    if reason == "aws_profile_not_found":
        return "AWS profile not found on the Studio backend host. Create it there or choose another profile."
    if reason == "aws_credentials_missing":
        return "No AWS credentials are available for the selected profile. Configure it on the Studio backend host."
    if isinstance(error, (SSOTokenLoadError, UnauthorizedSSOTokenError)):
        return "AWS SSO session is missing or expired. Run aws sso login --profile <profile> on the backend host."
    if reason == "aws_credentials_expired":
        return "AWS credentials have expired. Renew the selected profile on the Studio backend host, then check again."
    return "Cannot access AWS. Check profile credentials, network access, object read permissions, role trust, and URI."


def aws_auth_error(error: Exception, profile_name: str | None = None) -> CloudProviderError:
    """Describe recovery without exposing SDK details or guessing a profile's credential provider."""
    from botocore.exceptions import SSOTokenLoadError, UnauthorizedSSOTokenError

    if isinstance(error, IotCertificateError):
        return CloudProviderError(code=error.reason, title=error.title, detail=error.detail)
    reason = credential_error_reason(error)
    title = {
        "aws_profile_not_found": "AWS profile not found",
        "aws_credentials_missing": "AWS credentials missing",
        "aws_credentials_expired": "AWS credentials expired",
        "aws_access_denied": "AWS access denied",
    }.get(reason or "", "Cannot access AWS")
    detail = aws_auth_error_detail(error)
    command = None
    if profile_name and isinstance(error, (SSOTokenLoadError, UnauthorizedSSOTokenError)):
        title = "AWS SSO session missing or expired"
        detail = (
            f'Renew the session for profile "{profile_name}" on the machine running the Studio backend. '
            "After signing in, retry the operation in Studio."
        )
        command = shlex.join(["aws", "sso", "login", f"--profile={profile_name}"])
    return CloudProviderError(code=reason or "aws_unavailable", title=title, detail=detail, command=command)


class AwsCredentialProvider(ABC):
    """Strategy interface for acquiring AWS sessions and diagnosing authentication failures."""

    @abstractmethod
    def create_session(
        self,
        *,
        region: str | None = None,
        role_arn: str | None = None,
    ) -> Any:
        """Return an authenticated boto3.Session, optionally scoped to role_arn."""

    @abstractmethod
    def diagnose_error(self, error: Exception) -> AwsAuthenticationIssue | None:
        """Return a recoverable credential issue, excluding permission and network failures."""

    @abstractmethod
    def to_operational_error(self, error: Exception) -> OperationalIssue:
        """Translate a provider failure into the common public OperationalIssue contract."""


class ProfileCredentialProvider(AwsCredentialProvider):
    """Authenticate AWS operations using a backend-local named profile."""

    def __init__(self, profile_name: str) -> None:
        self.profile_name = profile_name

    def create_session(
        self,
        *,
        region: str | None = None,
        role_arn: str | None = None,
    ) -> Any:
        require_aws_feature_active()
        import boto3
        from botocore.config import Config

        source = boto3.Session(profile_name=self.profile_name)
        if role_arn is None:
            return source
        sts = source.client(
            "sts",
            region_name=region or source.region_name or "us-east-1",
            config=Config(connect_timeout=5, read_timeout=10),
        )
        creds = sts.assume_role(RoleArn=role_arn, RoleSessionName="physicalai-studio")["Credentials"]
        return boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=region,
        )

    def diagnose_error(self, error: Exception) -> AwsAuthenticationIssue | None:
        return authentication_issue(error, self.profile_name)

    def to_operational_error(self, error: Exception) -> OperationalIssue:
        return operational_aws_error(error, self.profile_name)


class IotCertificateCredentialProvider(AwsCredentialProvider):
    """Authenticate AWS operations by exchanging the robot's X.509 certificate via AWS IoT Core."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path

    def _load_config(self) -> GreengrassDeviceConfig:
        path = self.config_path or get_settings().greengrass_config_path
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise IotCertificateError(
                reason="aws_credentials_missing",
                title="Robot certificate configuration missing",
                detail="Cannot read the Greengrass configuration file on the Studio backend host.",
            ) from exc
        except yaml.YAMLError as exc:
            raise IotCertificateError(
                reason="aws_credentials_missing",
                title="Invalid robot certificate configuration",
                detail="The Greengrass configuration file on the Studio backend host is not valid YAML.",
            ) from exc
        if not isinstance(raw, dict):
            raise IotCertificateError(
                reason="aws_credentials_missing",
                title="Invalid robot certificate configuration",
                detail="The Greengrass configuration file on the Studio backend host is invalid.",
            )
        raw_system = raw.get("system")
        system: dict[str, Any] = raw_system if isinstance(raw_system, dict) else {}
        raw_services = raw.get("services")
        services: dict[str, Any] = raw_services if isinstance(raw_services, dict) else {}
        raw_nucleus = services.get("aws.greengrass.Nucleus")
        nucleus: dict[str, Any] = raw_nucleus if isinstance(raw_nucleus, dict) else {}
        raw_cfg = nucleus.get("configuration")
        nucleus_cfg: dict[str, Any] = raw_cfg if isinstance(raw_cfg, dict) else {}
        base_dir = path.parent
        cert_path = Path(system.get("certificateFilePath") or (base_dir / "device.pem.crt"))
        key_path = Path(system.get("privateKeyPath") or (base_dir / "private.pem.key"))
        ca_path = Path(system.get("rootCaPath") or (base_dir / "AmazonRootCA1.pem"))
        thing_name = system.get("thingName")
        cred_endpoint = nucleus_cfg.get("iotCredEndpoint")
        if not isinstance(thing_name, str) or not thing_name.strip():
            raise IotCertificateError(
                reason="aws_credentials_missing",
                title="Robot thing name missing",
                detail="The Greengrass configuration is missing system.thingName.",
            )
        if not isinstance(cred_endpoint, str) or not cred_endpoint.strip():
            raise IotCertificateError(
                reason="aws_credentials_missing",
                title="IoT credential endpoint missing",
                detail="The Greengrass configuration is missing iotCredEndpoint.",
            )
        for file_path, label in (
            (cert_path, "device certificate"),
            (key_path, "private key"),
            (ca_path, "root CA certificate"),
        ):
            if not file_path.is_file():
                raise IotCertificateError(
                    reason="aws_credentials_missing",
                    title="Robot certificate file missing",
                    detail=f"The robot {label} file was not found at the configured path on the backend host.",
                )
        role_alias = nucleus_cfg.get("iotRoleAlias")
        aws_region = nucleus_cfg.get("awsRegion")
        return GreengrassDeviceConfig(
            certificate_file_path=cert_path,
            private_key_path=key_path,
            root_ca_path=ca_path,
            thing_name=thing_name.strip(),
            iot_cred_endpoint=cred_endpoint.strip(),
            iot_role_alias=role_alias.strip() if isinstance(role_alias, str) and role_alias.strip() else None,
            aws_region=aws_region.strip() if isinstance(aws_region, str) and aws_region.strip() else None,
        )

    @staticmethod
    def _resolve_role_alias(config: GreengrassDeviceConfig, role_arn: str | None) -> str:
        if role_arn:
            role_name = role_arn.rsplit("/", 1)[-1]
            if role_name.startswith("pai-studio-"):
                return role_name
        if config.iot_role_alias:
            return config.iot_role_alias
        if role_arn:
            return role_arn.rsplit("/", 1)[-1]
        raise IotCertificateError(
            reason="aws_credentials_missing",
            title="IoT role alias missing",
            detail="No IoT role alias is configured for robot certificate authentication.",
        )

    def create_session(
        self,
        *,
        region: str | None = None,
        role_arn: str | None = None,
    ) -> Any:
        require_aws_feature_active()
        import boto3

        config = self._load_config()
        role_alias = self._resolve_role_alias(config, role_arn)
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(config.root_ca_path))
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.load_cert_chain(
                certfile=str(config.certificate_file_path),
                keyfile=str(config.private_key_path),
            )
        except (OSError, ssl.SSLError) as exc:
            raise IotCertificateError(
                reason="aws_credentials_missing",
                title="Invalid robot certificate or key",
                detail="Cannot load the robot X.509 certificate or private key on the Studio backend host.",
            ) from exc

        url = f"https://{config.iot_cred_endpoint}/role-aliases/{role_alias}/credentials"
        try:
            with httpx.Client(
                verify=ssl_context,
                timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = client.get(url, headers={"x-amzn-iot-thingname": config.thing_name})
        except ssl.SSLError as exc:
            raise IotCertificateError(
                reason="aws_credentials_expired",
                title="Robot certificate rejected",
                detail="TLS handshake with AWS IoT Core failed. Verify the robot certificate is valid and active.",
            ) from exc
        except httpx.HTTPError as exc:
            raise IotCertificateError(
                reason="aws_unavailable",
                title="Cannot reach AWS IoT Core",
                detail="Failed to contact the AWS IoT credential provider endpoint. Check network connectivity.",
            ) from exc

        if response.status_code in {401, 403}:
            raise IotCertificateError(
                reason="aws_credentials_expired",
                title="Robot certificate rejected",
                detail=(
                    f'AWS IoT Core rejected certificate authentication for thing "{config.thing_name}". '
                    "Verify the certificate is ACTIVE and attached to the IoT policy and role alias."
                ),
            )
        if response.status_code >= 400:
            raise IotCertificateError(
                reason="aws_unavailable",
                title="AWS IoT credential exchange failed",
                detail=f"AWS IoT credential endpoint returned HTTP {response.status_code}.",
            )
        try:
            payload = response.json()
            creds = payload["credentials"]
            access_key = creds["accessKeyId"]
            secret_key = creds["secretAccessKey"]
            session_token = creds["sessionToken"]
        except (ValueError, KeyError, TypeError) as exc:
            raise IotCertificateError(
                reason="aws_unavailable",
                title="Invalid AWS IoT credential response",
                detail="AWS IoT credential endpoint returned an unexpected payload.",
            ) from exc

        return boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region or config.aws_region or "us-east-1",
        )

    def diagnose_error(self, error: Exception) -> AwsAuthenticationIssue | None:
        issue = authentication_issue(error, "iot-certificate")
        if issue is not None and not isinstance(error, IotCertificateError):
            return issue.model_copy(
                update={
                    "title": "Robot AWS credentials expired",
                    "detail": (
                        "Temporary credentials from the robot certificate have expired or were revoked. "
                        "Check the robot certificate status in AWS IoT Core."
                    ),
                    "command": None,
                }
            )
        return issue

    def to_operational_error(self, error: Exception) -> OperationalIssue:
        if issue := self.diagnose_error(error):
            return operational_authentication_issue(issue)
        recovery = aws_auth_error(error)
        return OperationalIssue(
            id=str(uuid5(NAMESPACE_URL, f"studio:aws:{recovery.code}:iot-certificate")),
            title=recovery.title or "Cannot access provider",
            detail=recovery.detail,
            copy_text=recovery.command,
        )


def get_credential_provider(
    source: AwsCredentialSelection | AwsBatchConnection | str | None = None,
) -> AwsCredentialProvider:
    """Resolve the credential provider strategy for the given configuration or connection."""
    if is_robot_enrolled():
        return IotCertificateCredentialProvider()
    if isinstance(source, str):
        return ProfileCredentialProvider(source)
    if source is not None and source.profile_name is not None:
        return ProfileCredentialProvider(source.profile_name)
    return IotCertificateCredentialProvider()


def check_credentials(selection: AwsCredentialSelection) -> AwsIdentity:
    """Validate the selected named profile without changing AWS configuration."""
    from botocore.config import Config

    session = get_credential_provider(selection).create_session()
    identity = session.client(
        "sts", region_name=session.region_name or "us-east-1", config=Config(connect_timeout=5, read_timeout=10)
    ).get_caller_identity()
    return AwsIdentity(account=identity["Account"], arn=identity["Arn"])


def authentication_issue(error: Exception, profile_name: str) -> AwsAuthenticationIssue | None:
    """Return recoverable credential issues, excluding permission and network failures."""
    seen: set[int] = set()
    while credential_error_reason(error) is None and (nested := error.__cause__ or error.__context__):
        if not isinstance(nested, Exception):
            break
        if id(nested) in seen:
            break
        seen.add(id(error))
        error = nested
    if credential_error_reason(error) not in {
        "aws_credentials_expired",
        "aws_credentials_missing",
        "aws_profile_not_found",
    }:
        return None
    recovery = aws_auth_error(error, profile_name)
    if recovery.code is None or recovery.title is None:
        raise ValueError("AWS authentication recovery requires a code and title")
    if recovery.command is None and not isinstance(error, IotCertificateError):
        recovery.detail = f'Profile "{profile_name}": {recovery.detail}'
    return AwsAuthenticationIssue(
        profile_name=profile_name,
        code=recovery.code,
        title=recovery.title,
        detail=recovery.detail,
        command=recovery.command,
    )


class AwsAuthenticationRequiredError(Exception):
    """Reject submission before a job is created when its profile needs authentication."""

    def __init__(self, issue: AwsAuthenticationIssue) -> None:
        self.issue = issue
        super().__init__(issue.detail)


def operational_authentication_issue(issue: AwsAuthenticationIssue) -> OperationalIssue:
    """Give all observers of a profile's credential failure the same issue identity."""
    return OperationalIssue(
        id=str(uuid5(NAMESPACE_URL, f"studio:aws:authentication:{issue.profile_name}")),
        title=issue.title,
        detail=issue.detail,
        copy_text=issue.command,
    )


def operational_aws_error(error: Exception, profile_name: str | None = None) -> OperationalIssue:
    """Translate provider request failures into the common public error contract."""
    if profile_name and (issue := authentication_issue(error, profile_name)):
        return operational_authentication_issue(issue)
    recovery = aws_auth_error(error, profile_name)
    return OperationalIssue(
        id=str(uuid5(NAMESPACE_URL, f"studio:aws:{recovery.code}:{profile_name}")),
        title=recovery.title or "Cannot access provider",
        detail=recovery.detail,
        copy_text=recovery.command,
    )


def _profiles() -> list[str]:
    import boto3

    return sorted(boto3.Session().available_profiles)


async def get_capabilities() -> AwsAuthCapabilities:
    """List profiles configured on the backend and detect enrolled robot certificates."""
    require_aws_feature_active()
    enrolled = await asyncio.to_thread(is_robot_enrolled)
    profiles = await asyncio.to_thread(_profiles)
    return AwsAuthCapabilities(profiles=profiles, use_device_certificate=enrolled)
