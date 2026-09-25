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

/**
 * Returns true when managed SSH trainers are unavailable
 * (HTTP 503, `{ error_code: "ssh_feature_unavailable" }`).
 *
 * The backend disables SSH access when bound to a non-loopback address.
 * Callers should hide SSH-only controls and keep direct trainers available.
 */
export const isSshFeatureUnavailableError = (error: unknown): boolean =>
    typeof error === 'object' &&
    error !== null &&
    'error_code' in error &&
    (error as Record<string, unknown>).error_code === 'ssh_feature_unavailable';

interface ApiErrorBody {
    error_code?: string;
    message?: string | Record<string, string[]>;
    http_status?: number;
}

// The app's own `exception_handlers.py` reshapes a body-validation failure
// into one of two shapes before it ever reaches the client - the raw FastAPI
// `{ detail: [...] }` default is never actually returned by this backend, but
// handled below too as a defensive fallback:
//   - `RequestValidationError` (a field failed its own Pydantic constraint,
//     e.g. a pattern mismatch) -> `validation_exception_handler` ->
//     `message` is a `Record<field, string[]>`, not a string (see
//     `ApiErrorBody.message` above).
//   - a `pydantic.ValidationError` raised directly by application code ->
//     `pydantic_validation_exception_handler` -> `{ errors: [{ message,
//     location }] }`, with no top-level `message` at all.
interface PydanticValidationErrorBody {
    errors?: { message?: string; location?: string }[];
}

interface ValidationErrorBody {
    detail?: { loc?: (string | number)[]; msg?: string }[];
}

const joinFieldMessages = (field: string | undefined, messages: string[]): string =>
    field !== undefined && field !== '' ? `${field}: ${messages.join(', ')}` : messages.join(', ');

/**
 * Extracts the human-readable message from a backend error response.
 *
 * Tries, in order: a plain string `message`; the app's reshaped field-level
 * validation error (`message` as `Record<field, string[]>`); the app's
 * reshaped pydantic validation error (`errors: [{ message, location }]`);
 * FastAPI's raw `{ detail: [...] }` default, for a request that somehow
 * never reached this app's own handlers. Multiple field errors are joined
 * with '; '. Returns undefined when nothing readable is found.
 */
export const getApiErrorMessage = (error: unknown): string | undefined => {
    if (typeof error !== 'object' || error === null) {
        return undefined;
    }
    if ('message' in error) {
        const { message } = error as ApiErrorBody;
        if (typeof message === 'string' && message !== '') {
            return message;
        }
        if (typeof message === 'object' && message !== null) {
            const fieldMessages = Object.entries(message)
                .filter((entry): entry is [string, string[]] => Array.isArray(entry[1]))
                .map(([field, messages]) => joinFieldMessages(field, messages));
            if (fieldMessages.length > 0) {
                return fieldMessages.join('; ');
            }
        }
    }
    if ('errors' in error) {
        const { errors } = error as PydanticValidationErrorBody;
        const messages = (Array.isArray(errors) ? errors : [])
            .filter(
                (item): item is { message: string; location?: string } =>
                    item !== null && typeof item === 'object' && typeof item.message === 'string'
            )
            .map((item) => joinFieldMessages(item.location, [item.message]));
        if (messages.length > 0) {
            return messages.join('; ');
        }
    }
    if ('detail' in error) {
        const { detail } = error as ValidationErrorBody;
        const messages = (Array.isArray(detail) ? detail : [])
            .map((item) => {
                const field = Array.isArray(item?.loc) ? item.loc.at(-1) : undefined;
                return typeof item?.msg === 'string' && item.msg !== ''
                    ? joinFieldMessages(typeof field === 'string' ? field : undefined, [item.msg])
                    : undefined;
            })
            .filter((msg): msg is string => msg !== undefined);
        if (messages.length > 0) {
            return messages.join('; ');
        }
    }
    return undefined;
};

export const getSshHostKeyFingerprint = (error: unknown): string | undefined => {
    if (typeof error !== 'object' || error === null) {
        return undefined;
    }
    const payload = error as Record<string, unknown>;
    return payload.error_code === 'ssh_host_key_confirmation_required' && typeof payload.fingerprint === 'string'
        ? payload.fingerprint
        : undefined;
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
