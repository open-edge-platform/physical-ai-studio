import { useState } from 'react';

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { render } from '../../../../../test-utils/render';
import { FieldSchema } from '../types';
import {
    CalibrationField,
    isExpectedType,
    validateCalibrationEntry,
    validateCalibrationPayload,
} from './calibration-field';

const jointCalibrationSchema: FieldSchema = {
    type: 'object',
    properties: {
        id: { type: 'integer' },
        drive_mode: { type: 'integer' },
        homing_offset: { type: 'integer' },
        range_min: { type: 'integer' },
        range_max: { type: 'integer' },
    },
    required: ['id', 'drive_mode', 'homing_offset', 'range_min', 'range_max'],
};

const ControlledCalibrationField = ({
    initialValue = null,
    isRequired = false,
    onChange,
}: {
    initialValue?: unknown;
    isRequired?: boolean;
    onChange?: (value: unknown) => void;
}) => {
    const [value, setValue] = useState<unknown>(initialValue);

    return (
        <CalibrationField
            label='Calibration'
            description='Upload robot calibration values'
            value={value}
            isRequired={isRequired}
            valueSchema={jointCalibrationSchema}
            onChange={(next) => {
                setValue(next);
                onChange?.(next);
            }}
        />
    );
};

describe('CalibrationField', () => {
    it('validates primitive and object schema value types', () => {
        expect(isExpectedType(1, 'integer')).toBe(true);
        expect(isExpectedType(1.5, 'integer')).toBe(false);
        expect(isExpectedType(Number.NaN, 'number')).toBe(false);
        expect(isExpectedType('servo', 'string')).toBe(true);
        expect(isExpectedType(false, 'boolean')).toBe(true);
        expect(isExpectedType({}, 'object')).toBe(true);
        expect(isExpectedType([], 'object')).toBe(false);
        expect(isExpectedType(null, 'object')).toBe(false);
    });

    it('validates calibration entries against the value schema', () => {
        const validEntry = { id: 1, drive_mode: 0, homing_offset: 10, range_min: -100, range_max: 100 };

        expect(validateCalibrationEntry(validEntry, jointCalibrationSchema, {})).toBeNull();
        expect(validateCalibrationEntry([], jointCalibrationSchema, {})).toBe(
            'Each calibration entry must be a JSON object.'
        );
        expect(validateCalibrationEntry({ ...validEntry, id: null }, jointCalibrationSchema, {})).toBe(
            "Calibration entries must include 'id'."
        );
        expect(validateCalibrationEntry({ ...validEntry, id: '1' }, jointCalibrationSchema, {})).toBe(
            "Calibration field 'id' has an invalid value type."
        );
    });

    it('validates calibration payloads as joint-name keyed objects', () => {
        expect(validateCalibrationPayload([], jointCalibrationSchema, {})).toBe(
            'Calibration JSON must be an object keyed by joint name.'
        );
        expect(validateCalibrationPayload({ shoulder_pan: { id: 1 } }, undefined, {})).toBeNull();
    });

    it('shows optional marker when field is not required', () => {
        render(<ControlledCalibrationField isRequired={false} />);

        expect(screen.getByText('Calibration (optional)')).toBeVisible();
    });

    it('uploads calibration JSON, updates value, and shows sorted preview', async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();
        const calibrationPayload = {
            wrist_flex: { id: 5, drive_mode: 0, homing_offset: 11, range_min: -80, range_max: 80 },
            shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 10, range_min: -100, range_max: 100 },
            elbow_flex: { id: 3, drive_mode: 0, homing_offset: 12, range_min: -90, range_max: 90 },
        };
        const { container } = render(<ControlledCalibrationField onChange={onChange} />);
        const fileInput = container.querySelector('input[type="file"]');

        expect(fileInput).not.toBeNull();
        if (fileInput === null) {
            throw new Error('Expected calibration file input to be rendered.');
        }

        await user.upload(
            fileInput as HTMLInputElement,
            new File([JSON.stringify(calibrationPayload)], 'calibration.json', { type: 'application/json' })
        );

        expect(onChange).toHaveBeenCalledWith(calibrationPayload);
        expect(screen.getByRole('table', { name: 'Calibration preview' })).toBeVisible();

        const rows = screen.getAllByRole('row');
        expect(rows[1]).toHaveTextContent('shoulder_pan');
        expect(rows[2]).toHaveTextContent('elbow_flex');
        expect(rows[3]).toHaveTextContent('wrist_flex');
    });

    it('shows an error for invalid JSON upload', async () => {
        const user = userEvent.setup();
        const { container } = render(<ControlledCalibrationField />);
        const fileInput = container.querySelector('input[type="file"]');

        expect(fileInput).not.toBeNull();
        if (fileInput === null) {
            throw new Error('Expected calibration file input to be rendered.');
        }

        await user.upload(
            fileInput as HTMLInputElement,
            new File(['{"bad_json":'], 'broken.json', { type: 'application/json' })
        );

        expect(await screen.findByText('Could not parse JSON. Upload a valid calibration .json file.')).toBeVisible();
    });

    it('clears calibration to an empty object', async () => {
        const onChange = vi.fn();
        const user = userEvent.setup();

        render(<ControlledCalibrationField initialValue={{ shoulder_pan: { id: 1 } }} onChange={onChange} />);

        await user.click(screen.getByRole('button', { name: 'Clear' }));

        expect(onChange).toHaveBeenCalledWith({});
    });

    it('renders contextual help and learn more link when info is provided', async () => {
        const user = userEvent.setup();

        render(
            <CalibrationField
                label='Calibration'
                description='Upload robot calibration values'
                isRequired={false}
                value={null}
                valueSchema={jointCalibrationSchema}
                info={{
                    title: 'Calibration JSON',
                    description: 'Use calibration exported from the control board tools.',
                    link_url: 'https://example.com/calibration-docs',
                    variant: 'help',
                }}
                onChange={() => undefined}
            />
        );

        await user.click(screen.getByRole('button', { name: /Help$/ }));

        expect(await screen.findByRole('heading', { name: 'Calibration JSON' })).toBeVisible();
        expect(screen.getByText('Use calibration exported from the control board tools.')).toBeVisible();
        expect(screen.getByRole('link', { name: 'Learn more' })).toHaveAttribute(
            'href',
            'https://example.com/calibration-docs'
        );
    });
});
