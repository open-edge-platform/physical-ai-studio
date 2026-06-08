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
