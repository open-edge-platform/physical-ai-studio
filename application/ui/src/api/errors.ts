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
