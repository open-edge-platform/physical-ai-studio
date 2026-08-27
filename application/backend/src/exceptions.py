import http
from enum import StrEnum
from uuid import UUID


class ResourceType(StrEnum):
    """Enumeration for resource types."""

    PROJECT = "Project"
    ROBOT = "Robot"
    CAMERA = "Camera"
    ENVIRONMENT = "Environment"
    DATASET = "Dataset"
    MODEL = "Model"
    REMOTE_TRAINER = "Remote trainer"
    REMOTE_SERVER = "Remote server"
    JOB = "JOB"
    JOB_FILE = "JOB_FILE"


class BaseException(Exception):
    """
    Base class for PhysicalAI exceptions with a predefined HTTP error code.

    :param message: str message providing short description of error
    :param error_code: str id of error
    :param http_status: int default http status code to return to user
    """

    def __init__(self, message: str, error_code: str, http_status: int, *, phase: str | None = None) -> None:
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        self.phase = phase
        super().__init__(message)


class ResourceNotFoundError(BaseException):
    """
    Exception raised when a resource could not be found in database.

    :param resource_id: ID of the resource that was not found
    """

    def __init__(self, resource_type: ResourceType, resource_id: str | UUID, message: str | None = None):
        msg = (
            message or f"The requested {resource_type} could not be found. {resource_type.title()} ID: `{resource_id}`."
        )

        super().__init__(
            message=msg,
            error_code=f"{resource_type}_not_found",
            http_status=http.HTTPStatus.NOT_FOUND,
        )


class DuplicateJobException(BaseException):
    """
    Exception raised when attempting to submit a duplicate job.

    :param message: str containing a custom message about the duplicate job.
    """

    def __init__(self, message: str = "A job with the same payload is already running or queued") -> None:
        super().__init__(message=message, error_code="duplicate_job", http_status=http.HTTPStatus.CONFLICT)


class ResourceInUseError(BaseException):
    """Exception raised when trying to delete a resource that is currently in use."""

    def __init__(self, resource_type: ResourceType, resource_id: str | UUID, message: str | None = None):
        msg = message or f"{resource_type} with ID {resource_id} cannot be deleted because it is in use."
        super().__init__(
            message=msg,
            error_code=f"{resource_type}_in_use",
            http_status=http.HTTPStatus.CONFLICT,
        )


class RobotPluginUnavailableError(BaseException):
    """Raised when a robot's catalog plugin is not installed."""

    def __init__(self, robot_name: str, robot_type: str) -> None:
        super().__init__(
            message=(
                f"Robot '{robot_name}' requires unavailable plugin type '{robot_type}'. "
                "Reinstall the plugin before connecting."
            ),
            error_code="robot_plugin_unavailable",
            http_status=http.HTTPStatus.CONFLICT,
        )


class ResourceAlreadyExistsError(BaseException):
    """
    Exception raised when a resource already exists.

    :param resource_name: Name of the resource that was not found
    """

    def __init__(self, resource_name: str, detail: str) -> None:
        super().__init__(
            message=f"{resource_name} already exists. {detail}",
            error_code=f"{resource_name}_already_exists",
            http_status=http.HTTPStatus.CONFLICT,
        )


class UnsupportedDeviceError(BaseException):
    """Exception raised when a requested training device is not available on the system."""

    def __init__(self, device_type: str, supported: list[str]) -> None:
        supported_str = ", ".join(supported) if supported else "none"
        super().__init__(
            message=f"Device type '{device_type}' is not available for training. Supported devices: {supported_str}.",
            error_code="unsupported_device",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class RemoteResumeUnsupportedError(BaseException):
    """Raised when a job would resume from a base model on a remote trainer.

    Resuming needs the base model's checkpoint, and the trainer protocol has no
    way to send one: the only upload endpoint takes the dataset. Rejecting the
    submission is better than accepting it and silently training from scratch.
    """

    def __init__(self) -> None:
        super().__init__(
            message=(
                "Continuing training from an existing model is only supported on this machine. "
                "Select local training, or start a new model on the remote trainer."
            ),
            error_code="remote_resume_unsupported",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class RemoteServerNotReadyError(BaseException):
    """Raised when an SSH job targets a server that has not passed preflight.

    Studio never dials SSH from job submission (only the explicit save/check
    actions do); this only consults the server's persisted last-check summary.
    """

    def __init__(self, server_name: str, last_check_status: str) -> None:
        super().__init__(
            message=(
                f"Remote server '{server_name}' is not ready for training "
                f"(last check status: {last_check_status}). Verify the server before submitting a job."
            ),
            error_code="remote_server_not_ready",
            http_status=http.HTTPStatus.CONFLICT,
        )


class RemoteServerAliasNotFoundError(BaseException):
    """Raised when an SSH job's server names an SSH alias no longer in the config.

    Unlike `RemoteServerNotReadyError`, this is checked by parsing the SSH
    config file directly (no SSH dial), so it also catches a `Host` entry that
    was renamed or removed since the server was last verified.
    """

    def __init__(self, server_name: str, ssh_host_alias: str) -> None:
        super().__init__(
            message=(
                f"Remote server '{server_name}' points at SSH host alias '{ssh_host_alias}', "
                "which is no longer in your SSH config. Restore the Host entry, or edit the server."
            ),
            error_code="remote_server_alias_not_found",
            http_status=http.HTTPStatus.CONFLICT,
        )


class InvalidJobStateError(BaseException):
    """Raised when a job action is not valid in the current state."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="invalid_job_state",
            http_status=http.HTTPStatus.CONFLICT,
        )


class DuplicateImportSourceError(BaseException):
    """Raised when importing an already imported source UUID."""

    def __init__(self, resource_kind: str, source_uuid: str) -> None:
        super().__init__(
            message=f"{resource_kind} with original source UUID `{source_uuid}` was already imported.",
            error_code="duplicate_import_source",
            http_status=http.HTTPStatus.CONFLICT,
        )


class ZipBombDetectedError(BaseException):
    """Raised when an uploaded archive is considered unsafe."""

    def __init__(self, message: str = "Uploaded archive was rejected by zip safety validation") -> None:
        super().__init__(
            message=message,
            error_code="zip_bomb_detected",
            http_status=http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class InvalidArchiveError(BaseException):
    """Raised when an uploaded archive is invalid or unreadable."""

    def __init__(self, message: str = "Uploaded archive is invalid or unreadable") -> None:
        super().__init__(
            message=message,
            error_code="invalid_archive",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class UploadTooLargeError(BaseException):
    """Raised when the HTTP upload exceeds the configured maximum size."""

    def __init__(self, message: str = "Uploaded file exceeds the maximum allowed size") -> None:
        super().__init__(
            message=message,
            error_code="upload_too_large",
            http_status=http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class InvalidResourceError(BaseException):
    """
    Exception raised when a resource is not what was expected.

    :param resource_name: Name of the resource that was not found
    """

    def __init__(self, resource_name: str, detail: str) -> None:
        super().__init__(
            message=f"{resource_name} invalid resource. {detail}",
            error_code=f"{resource_name}_invalid_resource",
            http_status=http.HTTPStatus.CONFLICT,
        )


class InsufficientDiskSpaceError(BaseException):
    """Raised when there is not enough free disk space to safely store the upload or extraction."""

    def __init__(self, message: str = "Insufficient disk space to process the upload") -> None:
        super().__init__(
            message=message,
            error_code="insufficient_disk_space",
            http_status=http.HTTPStatus.INSUFFICIENT_STORAGE,
        )


class RecordingLockError(BaseException):
    """Raised when a camera cannot be modified because it is locked by an active recording session."""

    def __init__(self, message: str = "Camera is in use by an active recording session.") -> None:
        super().__init__(
            message=message,
            error_code="recording_locked",
            http_status=http.HTTPStatus.LOCKED,
        )


class CameraSettingsConflictError(BaseException):
    """Raised when a session asks for camera settings that another project has pinned."""

    def __init__(
        self,
        *,
        project_name: str,
        pinned: tuple[int, int, int],
        requested: tuple[int, int, int],
    ) -> None:
        width, height, fps = pinned
        req_width, req_height, req_fps = requested
        super().__init__(
            message=(
                f"Camera is already in use by project {project_name!r} at {width}x{height}@{fps}. "
                f"This session requested {req_width}x{req_height}@{req_fps}."
            ),
            error_code="camera_settings_conflict",
            http_status=http.HTTPStatus.LOCKED,
        )


class RuntimeSessionBusyError(BaseException):
    """Raised when a live runtime session already holds the robot being asked for."""

    def __init__(self, *, robot_name: str | None = None, pid: int | None = None) -> None:
        subject = f"Robot {robot_name!r} is" if robot_name else "This robot is"
        holder = f" (pid {pid})" if pid is not None else ""
        message = (
            f"{subject} already in use by a running session{holder}. "
            "Stop that session, or wait for it to disconnect, then try again."
        )
        super().__init__(
            message=message,
            error_code="runtime_session_busy",
            http_status=http.HTTPStatus.LOCKED,
        )


class RobotDeviceAlreadyOwnedError(BaseException):
    """Raised when a SharedRobot device is already locked under another session name."""

    def __init__(self, *, device_ids: tuple[str, ...] | None = None) -> None:
        if device_ids:
            devices = ", ".join(device_ids)
            message = (
                f"Device {devices} is already in use by another session. "
                "Stop the other session or wait for it to disconnect, then try again."
            )
        else:
            message = (
                "This robot device is already in use by another session. "
                "Stop the other session or wait for it to disconnect, then try again."
            )
        super().__init__(
            message=message,
            error_code="robot_device_already_owned",
            http_status=http.HTTPStatus.CONFLICT,
        )


class RobotNameConflictError(BaseException):
    """Raised when a SharedRobot name is claimed for different devices."""

    def __init__(self, *, robot_name: str | None = None) -> None:
        # The transport name is the robot's id, not its display name, so a
        # conflict means this robot already has a session bound to different
        # hardware than the one it now resolves to.
        subject = f"Robot {robot_name!r} is" if robot_name else "This robot is"
        message = (
            f"{subject} already running in another session that is bound to a different device. "
            "Stop that session, or check that this robot still points at the right hardware, then try again."
        )
        super().__init__(
            message=message,
            error_code="robot_name_conflict",
            http_status=http.HTTPStatus.CONFLICT,
        )


class RobotProtocolMismatchError(BaseException):
    """Raised when an existing SharedRobot owner speaks an unsupported protocol version."""

    def __init__(
        self,
        message: str = (
            "An existing robot session uses an incompatible software version. Restart all robot sessions and try again."
        ),
    ) -> None:
        super().__init__(
            message=message,
            error_code="robot_protocol_mismatch",
            http_status=http.HTTPStatus.CONFLICT,
        )


class ModelCameraMismatchError(BaseException):
    """Raised when a model's image inputs do not match the session's cameras."""

    def __init__(self, *, expected: list[str], provided: list[str]) -> None:
        expected_text = _format_camera_keys(expected)
        provided_text = _format_camera_keys(provided)
        message = (
            f"This model expects camera inputs {expected_text}, but this environment "
            f"provides {provided_text}. Cameras were probably renamed after the model "
            "was trained. Rename them back, or retrain."
        )
        super().__init__(
            message=message,
            error_code="model_camera_mismatch",
            http_status=http.HTTPStatus.CONFLICT,
        )
        self.expected = expected
        self.provided = provided


def _format_camera_keys(keys: list[str]) -> str:
    if not keys:
        return "none"
    return ", ".join(f"`{key}`" for key in keys)


class SharedRobotTransportError(BaseException):
    """Raised when SharedRobot transport fails (spawn, handshake, or wire)."""

    def __init__(
        self,
        message: str = "Could not connect to the robot. Check the connection and try again.",
    ) -> None:
        super().__init__(
            message=message,
            error_code="robot_transport_error",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class RobotIdentifyError(BaseException):
    """Raised when visually identifying a robot fails during joint motion."""

    def __init__(
        self,
        message: str = (
            "Robot identify failed: a joint could not be moved safely. Power-cycle the robot and try again."
        ),
    ) -> None:
        super().__init__(
            message=message,
            error_code="robot_identify_error",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class SshHostAliasNotFoundError(BaseException):
    """Raised when a server's SSH host alias is absent from the user's SSH config.

    Distinct from a connection failure: nothing was dialed, because there was no
    host to dial. A wildcard-only match lands here too - a pattern stanza is not
    a usable target.
    """

    def __init__(self, alias: str) -> None:
        super().__init__(
            message=(
                f"SSH host alias '{alias}' was not found in your SSH config. "
                f"Add a Host entry named '{alias}' to ~/.ssh/config, then try again."
            ),
            error_code="ssh_host_alias_not_found",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class SshHostKeyUnknownError(BaseException):
    """Raised when the host is absent from ``known_hosts``.

    Fails closed. Studio neither pins nor writes host keys, so the recovery is
    for the user to accept the fingerprint themselves.
    """

    def __init__(self, alias: str) -> None:
        super().__init__(
            message=(
                f"The host key for '{alias}' has not been accepted yet. "
                f"Run `ssh {alias}` once and accept its fingerprint, then try again."
            ),
            error_code="ssh_host_key_unknown",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class SshHostKeyMismatchError(BaseException):
    """Raised when the host key differs from the one in ``known_hosts``.

    Treated as untrusted rather than as a stale entry: this is what a
    machine-in-the-middle looks like, and it is also what a legitimately
    rebuilt host looks like. Studio cannot tell them apart, so it refuses.
    """

    def __init__(self, alias: str) -> None:
        super().__init__(
            message=(
                f"The host key for '{alias}' does not match the one in your known_hosts file. "
                "The server may have been rebuilt, or the connection may be intercepted. "
                "Verify the new fingerprint out of band before updating known_hosts."
            ),
            error_code="ssh_host_key_mismatch",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class SshAgentRequiredError(BaseException):
    """Raised when the resolved identity is passphrase-protected and no agent can unlock it.

    Studio never prompts for or stores a passphrase, so an agent is the only way
    a protected key can be used.
    """

    def __init__(self, alias: str) -> None:
        super().__init__(
            message=(
                f"The SSH key for '{alias}' is passphrase-protected and no SSH agent is available. "
                f"Start an agent and run `ssh-add`, then try again."
            ),
            error_code="ssh_agent_required",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class SshAuthenticationError(BaseException):
    """Raised when the server rejected every identity the SSH config offered."""

    def __init__(self, alias: str) -> None:
        super().__init__(
            message=(
                f"Authentication failed for '{alias}'. Check that `ssh {alias}` works from a terminal, then try again."
            ),
            error_code="ssh_authentication_failed",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class SshConnectionError(BaseException):
    """Raised when the resolved host could not be reached.

    Carries no underlying exception text: a raw SSH error can contain resolved
    hostnames and key paths, which must not reach an API response.
    """

    def __init__(self, alias: str, reason: str | None = None) -> None:
        detail = f" ({reason})" if reason else ""
        super().__init__(
            message=f"Could not connect to '{alias}'{detail}. Check that the server is reachable.",
            error_code="ssh_connection_failed",
            http_status=http.HTTPStatus.BAD_GATEWAY,
        )


class RemoteServerPreflightError(BaseException):
    """Raised when a server's blocking Tier 1 checks fail on create or update.

    Carries the structured per-check results so the UI can show which check
    failed rather than only that something did.
    """

    def __init__(self, message: str, failures: list[str] | None = None) -> None:
        self.failures = failures or []
        super().__init__(
            message=message,
            error_code="remote_server_preflight_failed",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class TrainerImageResolutionError(BaseException):
    """Raised when the device-specific `protocol-<N>` trainer image cannot be resolved.

    There is deliberately no fallback tag: a `latest` fallback here (unlike
    Tier 1's advisory preflight check) would silently run a job against an
    image whose protocol compatibility was never established.
    """

    def __init__(self, image_ref: str, protocol_version: int, detail: str | None = None) -> None:
        extra = f" ({detail})" if detail else ""
        super().__init__(
            message=(
                f"Could not resolve trainer image '{image_ref}' for protocol version {protocol_version}{extra}. "
                f"A matching `protocol-{protocol_version}`-tagged trainer image must be published before this "
                "job can run."
            ),
            error_code="trainer_image_unresolved",
            http_status=http.HTTPStatus.CONFLICT,
        )


class TrainerImagePullError(BaseException):
    """Raised when `docker pull` of the resolved digest failed on the remote host."""

    def __init__(self, image_ref: str, detail: str | None = None) -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            message=f"Could not pull trainer image '{image_ref}'{extra}.",
            error_code="trainer_image_pull_failed",
            http_status=http.HTTPStatus.BAD_GATEWAY,
        )


class TrainerImageVerificationError(BaseException):
    """Raised when the trainer image's signature could not be verified.

    Fails closed by default: this covers both a failed `cosign verify` and
    `cosign` being unavailable on the remote host. `cosign` being unavailable
    can be downgraded to a non-blocking warning via
    `Settings.ssh_require_cosign_verification`; a failed `cosign verify`
    always raises.
    """

    def __init__(self, image_ref: str, reason: str) -> None:
        super().__init__(
            message=f"Could not verify the signature of trainer image '{image_ref}': {reason}.",
            error_code="trainer_image_verification_failed",
            http_status=http.HTTPStatus.CONFLICT,
        )


class TrainerLibraryVersionError(BaseException):
    """Raised when the registry-reported `physicalai-train` version is below policy.

    Read from the registry manifest label before any pull, so a version-policy
    rejection never costs a multi-gigabyte transfer.
    """

    def __init__(self, policy_name: str, required_version: str, reported_version: str) -> None:
        super().__init__(
            message=(
                f"Trainer image reports physicalai-train version '{reported_version}', which does not meet "
                f"the '{policy_name}' policy's minimum of '{required_version}'."
            ),
            error_code="trainer_library_version_unmet",
            http_status=http.HTTPStatus.CONFLICT,
        )


class TrainerLibraryVersionMismatchError(BaseException):
    """Raised when the launched container's `/health` disagrees with the registry label.

    Defense in depth: the registry-manifest label is read before the pull, and
    this re-confirms it against the running container's own report.
    """

    def __init__(self, label_version: str, health_version: str) -> None:
        super().__init__(
            message=(
                f"Trainer image's registry label reports physicalai-train version '{label_version}', but the "
                f"running container's /health reports '{health_version}'."
            ),
            error_code="trainer_library_version_mismatch",
            http_status=http.HTTPStatus.CONFLICT,
        )


class GpuBusyTimeoutError(BaseException):
    """Raised when a remote GPU stayed busy past the configured give-up timeout."""

    def __init__(self, server_name: str, waited_s: float) -> None:
        super().__init__(
            message=f"GPU on remote server '{server_name}' stayed busy for {waited_s:.0f}s; giving up.",
            error_code="gpu_busy_timeout",
            http_status=http.HTTPStatus.CONFLICT,
        )


class RemoteDiskSpaceError(BaseException):
    """Raised when a remote server lacks room for this job's actual snapshot."""

    def __init__(self, server_name: str, free_bytes: int, required_bytes: int) -> None:
        super().__init__(
            message=(
                f"Remote server '{server_name}' has {free_bytes / (1024**3):.1f} GiB free, "
                f"but this job needs {required_bytes / (1024**3):.1f} GiB."
            ),
            error_code="remote_disk_insufficient",
            http_status=http.HTTPStatus.CONFLICT,
        )


class TrainerContainerLaunchError(BaseException):
    """Raised when the trainer container could not be started on the remote host."""

    def __init__(self, server_name: str, detail: str | None = None) -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            message=f"Could not start the trainer container on remote server '{server_name}'{extra}.",
            error_code="trainer_container_launch_failed",
            http_status=http.HTTPStatus.BAD_GATEWAY,
        )


class TrainerReadinessTimeoutError(BaseException):
    """Raised when the launched trainer never became ready, or reported no protocol version."""

    def __init__(self, server_name: str, detail: str | None = None) -> None:
        extra = f": {detail}" if detail else ""
        super().__init__(
            message=f"Trainer on remote server '{server_name}' did not become ready in time{extra}.",
            error_code="trainer_readiness_timeout",
            http_status=http.HTTPStatus.BAD_GATEWAY,
        )


class TrainerProtocolVersionMismatchError(BaseException):
    """Raised when the launched trainer's reported protocol version does not match.

    Strict for SSH-provisioned trainers: unlike Tier 1's advisory preflight,
    provisioning a real job never proceeds on a protocol mismatch.
    """

    def __init__(self, server_name: str, expected: int, reported: int | None) -> None:
        reported_text = str(reported) if reported is not None else "none"
        super().__init__(
            message=(
                f"Trainer on remote server '{server_name}' reports protocol version {reported_text}, "
                f"expected {expected}."
            ),
            error_code="trainer_protocol_mismatch",
            http_status=http.HTTPStatus.CONFLICT,
        )
