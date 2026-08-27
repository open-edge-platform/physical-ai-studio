/**
 * Returns true when the API error was caused by a recording lock (HTTP 423).
 *
 * The backend returns `{ error_code: "recording_locked", ... }` when a camera
 * is in use by an active recording session.
 */
export const isRecordingLockedError = (error: unknown): boolean =>
    typeof error === 'object' &&
    error !== null &&
    'error_code' in error &&
    (error as Record<string, unknown>).error_code === 'recording_locked';

/**
 * Returns true when the API error is a "resource in use" conflict (HTTP 409).
 *
 * The backend returns `{ error_code: "<Resource>_in_use", ... }` when a robot or camera
 * cannot be deleted because an environment still references it. This is an expected,
 * recoverable state — not an application failure — so callers should surface it as info.
 */
export const isResourceInUseError = (error: unknown): boolean =>
    typeof error === 'object' &&
    error !== null &&
    'error_code' in error &&
    typeof (error as Record<string, unknown>).error_code === 'string' &&
    (error as Record<string, string>).error_code.toLowerCase().endsWith('_in_use');

/**
 * Returns true when the API error was a serial port permission failure (HTTP 403).
 *
 * The backend returns `{ error_code: "serial_permission_denied", ... }` when the
 * process cannot open the robot's serial device (e.g. missing `dialout` access).
 */
export const isSerialPermissionDeniedError = (error: unknown): boolean =>
    typeof error === 'object' &&
    error !== null &&
    typeof (error as Record<string, unknown>).error_code === 'string' &&
    (error as Record<string, string>).error_code.toLowerCase() === 'serial_permission_denied';

/**
 * Returns true when a live runtime session holds the robot (HTTP 423).
 *
 * Expected when deleting a robot that is still being driven, or when connecting
 * with a different rig (leader, fps) than the session that already owns it.
 */
export const isRuntimeSessionBusyError = (error: unknown): boolean =>
    typeof error === 'object' &&
    error !== null &&
    'error_code' in error &&
    (error as Record<string, unknown>).error_code === 'runtime_session_busy';

interface ApiErrorBody {
    error_code?: string;
    message?: string;
    http_status?: number;
}

/**
 * Extracts the human-readable `message` from a backend error response
 * (`{ error_code, message, http_status }`). Returns undefined when absent.
 */
export const getApiErrorMessage = (error: unknown): string | undefined => {
    if (typeof error === 'object' && error !== null && 'message' in error) {
        const { message } = error as ApiErrorBody;
        return typeof message === 'string' ? message : undefined;
    }
    return undefined;
};

/**
 * Short title for robot connection errors surfaced over WebSocket or API responses.
 */
export const getRobotConnectionErrorTitle = (errorCode: string | null): string => {
    switch (errorCode) {
        case 'robot_device_already_owned':
        case 'runtime_session_busy':
            return 'Robot already in use';
        case 'robot_name_conflict':
            return 'Robot name conflict';
        case 'robot_protocol_mismatch':
            return 'Incompatible robot session';
        case 'robot_transport_error':
        case 'robot_connection_failed':
            return 'Connection failed';
        case 'connection_closed':
            return 'Connection lost';
        default:
            return 'Connection error';
    }
};
