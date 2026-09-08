import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { render } from '../../test-utils/render';
import { asCalibrationRows, CalibrationTable, formatCalibrationCell } from './calibration-table';

describe('CalibrationTable', () => {
    it('sorts calibration rows by ID and joint name', () => {
        expect(
            asCalibrationRows({
                wrist_flex: { id: 5 },
                shoulder_pan: { id: 1 },
                elbow_flex: { id: 3 },
                gripper: {},
                wrist_roll: {},
            }).map(({ joint }) => joint)
        ).toEqual(['shoulder_pan', 'elbow_flex', 'wrist_flex', 'gripper', 'wrist_roll']);
    });

    it('formats only finite numbers and non-empty strings for cells', () => {
        expect(formatCalibrationCell(10)).toBe('10');
        expect(formatCalibrationCell('servo')).toBe('servo');
        expect(formatCalibrationCell(Number.NaN)).toBe('-');
        expect(formatCalibrationCell('')).toBe('-');
        expect(formatCalibrationCell(null)).toBe('-');
    });

    it('renders sorted values with a supplied accessible name', () => {
        render(
            <CalibrationTable
                ariaLabel='SO101 calibration'
                calibration={{
                    wrist_flex: { id: 5, drive_mode: 0, homing_offset: 11, range_min: -80, range_max: 80 },
                    shoulder_pan: { id: 1, drive_mode: 1, homing_offset: 10, range_min: -100, range_max: 100 },
                }}
            />
        );

        const table = screen.getByRole('table', { name: 'SO101 calibration' });
        const rows = within(table).getAllByRole('row');

        expect(rows[1]).toHaveTextContent('shoulder_pan1110-100100');
        expect(rows[2]).toHaveTextContent('wrist_flex5011-8080');
    });

    it('renders only the requested columns and uses fallbacks for missing values', () => {
        render(
            <CalibrationTable
                calibration={{ shoulder_pan: { homing_offset: 10, range_min: null, range_max: Number.NaN } }}
                columns={['homing_offset', 'range_min', 'range_max']}
            />
        );

        const table = screen.getByRole('table', { name: 'Calibration values' });

        expect(within(table).getAllByRole('columnheader')).toHaveLength(4);
        expect(within(table).getByRole('columnheader', { name: 'Offset' })).toBeVisible();
        expect(within(table).queryByRole('columnheader', { name: 'ID' })).not.toBeInTheDocument();
        expect(
            within(table)
                .getAllByRole('cell')
                .map((cell) => cell.textContent)
        ).toEqual(['shoulder_pan', '10', '-', '-']);
    });
});
