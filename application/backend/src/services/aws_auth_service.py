"""Discover and validate backend-local AWS named profiles."""

import asyncio
import shlex
from uuid import NAMESPACE_URL, uuid5

from core.security import require_aws_feature_active
from schemas.aws_auth import AwsAuthCapabilities, AwsAuthenticationIssue, AwsCredentialSelection, AwsIdentity
from schemas.operational_issue import OperationalIssue
from schemas.remote_trainer import CloudProviderError


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

    if isinstance(error, ProfileNotFound):
        return "aws_profile_not_found"
    if isinstance(error, (NoCredentialsError, PartialCredentialsError)):
        return "aws_credentials_missing"
    if isinstance(
        error,
        (SSOTokenLoadError, UnauthorizedSSOTokenError, TokenRetrievalError, LoginRefreshRequired, LoginTokenLoadError),
    ):
        return "aws_credentials_expired"
    code = error.response.get("Error", {}).get("Code") if isinstance(error, ClientError) else None
    if code in {
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


def check_credentials(selection: AwsCredentialSelection) -> AwsIdentity:
    """Validate the selected named profile without changing AWS configuration."""
    require_aws_feature_active()
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=selection.profile_name)
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
    if recovery.command is None:
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
    """List profiles configured on the backend without authenticating."""
    require_aws_feature_active()
    return AwsAuthCapabilities(profiles=await asyncio.to_thread(_profiles))
