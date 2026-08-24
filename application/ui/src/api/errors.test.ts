import { describe, expect, it } from 'vitest';

import { getApiErrorMessage, isResourceInUseError, isSerialPermissionDeniedError } from './errors';

describe('isSerialPermissionDeniedError', () => {
    it('matches the serial_permission_denied code', () => {
        expect(
            isSerialPermissionDeniedError({
                error_code: 'serial_permission_denied',
                message: 'Permission denied while opening the serial device.',
                http_status: 403,
            })
        ).toBe(true);
    });

    it('matches case-insensitively', () => {
        expect(isSerialPermissionDeniedError({ error_code: 'SERIAL_PERMISSION_DENIED' })).toBe(true);
    });

    it('returns false for non-object, missing, or non-string error codes', () => {
        expect(isSerialPermissionDeniedError(null)).toBe(false);
        expect(isSerialPermissionDeniedError('serial_permission_denied')).toBe(false);
        expect(isSerialPermissionDeniedError({})).toBe(false);
        expect(isSerialPermissionDeniedError({ error_code: 403 })).toBe(false);
        expect(isSerialPermissionDeniedError({ error_code: 'robot_identify_error' })).toBe(false);
    });
});

describe('isResourceInUseError', () => {
    it('matches _in_use codes case-insensitively', () => {
        expect(isResourceInUseError({ error_code: 'robot_in_use' })).toBe(true);
        expect(isResourceInUseError({ error_code: 'ROBOT_IN_USE' })).toBe(true);
        expect(isResourceInUseError({ error_code: 'robot_identify_error' })).toBe(false);
    });
});

describe('getApiErrorMessage', () => {
    it('returns the message when present', () => {
        expect(getApiErrorMessage({ error_code: 'robot_identify_error', message: 'Identify failed.' })).toBe(
            'Identify failed.'
        );
    });

    it('returns undefined when the message is absent or not a string', () => {
        expect(getApiErrorMessage({})).toBeUndefined();
        expect(getApiErrorMessage({ message: 42 })).toBeUndefined();
        expect(getApiErrorMessage(null)).toBeUndefined();
    });
});
