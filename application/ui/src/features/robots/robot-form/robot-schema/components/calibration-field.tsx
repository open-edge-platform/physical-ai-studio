import { useState } from 'react';

import { Button, FileTrigger, Flex, Text, View } from '@geti-ui/ui';

import { CalibrationTable } from '../../../calibration-table';
import { InlineAlert } from '../../../setup-wizard/shared/inline-alert';
import { asRecord, resolveReference } from '../schema-utils';
import { FieldSchema } from '../types';

type CalibrationFieldProps = {
    label: string;
    description?: string;
    value: unknown;
    isRequired: boolean;
    onChange: (value: unknown) => void;
    valueSchema?: FieldSchema;
    definitions?: Record<string, FieldSchema>;
};

export const isExpectedType = (value: unknown, schemaType: string | undefined) => {
    if (schemaType === undefined) {
        return true;
    }
    if (schemaType === 'integer') {
        return typeof value === 'number' && Number.isInteger(value);
    }
    if (schemaType === 'number') {
        return typeof value === 'number' && Number.isFinite(value);
    }
    if (schemaType === 'string') {
        return typeof value === 'string';
    }
    if (schemaType === 'boolean') {
        return typeof value === 'boolean';
    }
    if (schemaType === 'object') {
        return typeof value === 'object' && value !== null && !Array.isArray(value);
    }
    return true;
};

export const validateCalibrationEntry = (
    entry: unknown,
    valueSchema: FieldSchema | undefined,
    definitions: Record<string, FieldSchema>
): string | null => {
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
        return 'Each calibration entry must be a JSON object.';
    }

    if (valueSchema === undefined) {
        return null;
    }

    const resolved = resolveReference(valueSchema, definitions);
    const entryRecord = asRecord(entry);
    const required = new Set(resolved.required ?? []);

    for (const requiredField of required) {
        if (entryRecord[requiredField] === undefined || entryRecord[requiredField] === null) {
            return `Calibration entries must include '${requiredField}'.`;
        }
    }

    for (const [name, fieldValue] of Object.entries(entryRecord)) {
        const fieldSchema = resolved.properties?.[name];
        if (fieldSchema === undefined) {
            continue;
        }
        const expectedSchema = resolveReference(fieldSchema, definitions);
        if (!isExpectedType(fieldValue, expectedSchema.type)) {
            return `Calibration field '${name}' has an invalid value type.`;
        }
    }

    return null;
};

export const validateCalibrationPayload = (
    value: unknown,
    valueSchema: FieldSchema | undefined,
    definitions: Record<string, FieldSchema>
): string | null => {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        return 'Calibration JSON must be an object keyed by joint name.';
    }

    for (const entry of Object.values(value)) {
        const error = validateCalibrationEntry(entry, valueSchema, definitions);
        if (error !== null) {
            return error;
        }
    }

    return null;
};

export const CalibrationField = ({
    label,
    description,
    value,
    isRequired,
    onChange,
    valueSchema,
    definitions = {},
}: CalibrationFieldProps) => {
    const [error, setError] = useState<string | null>(null);
    const calibration = asRecord(value);
    const hasCalibration = Object.keys(calibration).length > 0;

    const importCalibration = async (files: FileList | null) => {
        const file = files?.[0] ?? null;
        if (file === null) {
            return;
        }

        const text = await file.text();
        let parsed: unknown;
        try {
            parsed = JSON.parse(text);
        } catch {
            setError('Could not parse JSON. Upload a valid calibration .json file.');
            return;
        }

        const validationError = validateCalibrationPayload(parsed, valueSchema, definitions);
        if (validationError !== null) {
            setError(validationError);
            return;
        }

        setError(null);
        onChange(parsed);
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Text
                UNSAFE_style={{
                    fontSize: 'var(--spectrum-global-dimension-font-size-100)',
                    color: 'var(--spectrum-global-color-gray-800)',
                }}
            >
                {label}
                {isRequired ? ' *' : ' (optional)'}
            </Text>
            {description !== undefined && description !== '' && (
                <Text
                    UNSAFE_style={{
                        fontSize: 'var(--spectrum-global-dimension-font-size-100)',
                        color: 'var(--spectrum-global-color-gray-600)',
                    }}
                >
                    {description}
                </Text>
            )}
            <Flex gap='size-100' alignItems='center'>
                <FileTrigger acceptedFileTypes={['.json']} onSelect={importCalibration}>
                    <Button variant='secondary'>
                        {hasCalibration ? 'Replace calibration JSON' : 'Upload calibration JSON'}
                    </Button>
                </FileTrigger>
                {hasCalibration && (
                    <Button
                        variant='secondary'
                        onPress={() => {
                            setError(null);
                            onChange({});
                        }}
                    >
                        Clear
                    </Button>
                )}
            </Flex>
            {hasCalibration && (
                <View
                    borderColor='gray-300'
                    borderWidth='thin'
                    backgroundColor='gray-75'
                    padding='size-100'
                    UNSAFE_style={{ borderRadius: 'var(--spectrum-global-dimension-size-50)', overflowX: 'auto' }}
                >
                    <CalibrationTable calibration={calibration} ariaLabel='Calibration preview' />
                </View>
            )}
            {error !== null && <InlineAlert variant='error'>{error}</InlineAlert>}
        </Flex>
    );
};
