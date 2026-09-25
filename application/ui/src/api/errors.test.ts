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

    it('falls back to a FastAPI validation-error detail when message is absent', () => {
        expect(
            getApiErrorMessage({
                detail: [
                    {
                        loc: ['body', 'alias'],
                        msg: "String should match pattern '^[A-Za-z0-9]...'",
                        type: 'string_pattern_mismatch',
                    },
                ],
            })
        ).toBe("alias: String should match pattern '^[A-Za-z0-9]...'");
    });

    it('joins multiple validation-error details', () => {
        expect(
            getApiErrorMessage({
                detail: [
                    { loc: ['body', 'alias'], msg: 'Field required' },
                    { loc: ['body', 'hostname'], msg: 'Field required' },
                ],
            })
        ).toBe('alias: Field required; hostname: Field required');
    });

    it('omits the field prefix when loc has no field name', () => {
        expect(getApiErrorMessage({ detail: [{ loc: [], msg: 'Invalid request body' }] })).toBe('Invalid request body');
    });

    it('returns undefined for empty or malformed validation details', () => {
        expect(getApiErrorMessage({ detail: [] })).toBeUndefined();
        expect(getApiErrorMessage({ detail: 'Not found' })).toBeUndefined();
        expect(getApiErrorMessage({ detail: [{ loc: ['body'] }, null] })).toBeUndefined();
        expect(getApiErrorMessage({ errors: 'Not found' })).toBeUndefined();
    });

    it("reads the app's reshaped field-validation error (message as a field->messages record)", () => {
        // What `exception_handlers.validation_exception_handler` actually returns
        // for a `RequestValidationError` - e.g. a POST body failing a Pydantic
        // field constraint such as the SSH host alias pattern.
        expect(
            getApiErrorMessage({
                error_code: 'bad_request',
                message: { alias: ["String should match pattern '^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$'"] },
                http_status: 400,
            })
        ).toBe("alias: String should match pattern '^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$'");
    });

    it('joins multiple fields and multiple messages per field in a reshaped field-validation error', () => {
        expect(
            getApiErrorMessage({
                error_code: 'bad_request',
                message: { alias: ['Field required'], hostname: ['Field required', 'String too short'] },
                http_status: 400,
            })
        ).toBe('alias: Field required; hostname: Field required, String too short');
    });

    it("reads the app's reshaped pydantic validation error (errors array)", () => {
        // What `exception_handlers.pydantic_validation_exception_handler` returns
        // for a `pydantic.ValidationError` raised directly by application code.
        expect(
            getApiErrorMessage({
                error_code: 'invalid_payload',
                errors: [{ message: 'Field required', type: 'missing', location: 'alias' }],
                http_status: 400,
            })
        ).toBe('alias: Field required');
    });
});
