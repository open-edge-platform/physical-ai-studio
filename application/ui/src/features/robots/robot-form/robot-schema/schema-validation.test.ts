import { describe, expect, it } from 'vitest';

import { validateRequiredUiFields } from './schema-validation';

const schema = {
    type: 'object',
    properties: {
        id: {
            type: 'string',
            default: '',
            'x-physicalai-ui': { required: true },
        },
        optional: {
            type: 'string',
            default: '',
        },
    },
};

describe('validateRequiredUiFields', () => {
    it.each([undefined, '', '   '])('rejects an empty required string value: %j', (id) => {
        expect(validateRequiredUiFields(schema, { id })).toEqual(['id is required']);
    });

    it('accepts a non-empty required string value', () => {
        expect(validateRequiredUiFields(schema, { id: 'robot-1' })).toEqual([]);
    });

    it('does not change validation for fields without the extension', () => {
        expect(validateRequiredUiFields(schema, { id: 'robot-1', optional: '   ' })).toEqual([]);
    });
});
