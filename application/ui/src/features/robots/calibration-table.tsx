import { CSSProperties } from 'react';

type CalibrationEntry = {
    id?: unknown;
    drive_mode?: unknown;
    homing_offset?: unknown;
    range_min?: unknown;
    range_max?: unknown;
};

type CalibrationRow = {
    joint: string;
    value: CalibrationEntry;
};

export type CalibrationColumn = keyof CalibrationEntry;

type CalibrationTableProps = {
    calibration: Record<string, unknown>;
    columns?: readonly CalibrationColumn[];
    ariaLabel?: string;
    className?: string;
};

const columnLabels: Record<CalibrationColumn, string> = {
    id: 'ID',
    drive_mode: 'Drive',
    homing_offset: 'Offset',
    range_min: 'Min',
    range_max: 'Max',
};

const asRecord = (value: unknown): Record<string, unknown> =>
    typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

export const asCalibrationRows = (value: Record<string, unknown>): CalibrationRow[] =>
    Object.entries(value)
        .map(([joint, entry]) => ({ joint, value: asRecord(entry) }))
        .sort((left, right) => {
            const leftId =
                typeof left.value.id === 'number' && Number.isFinite(left.value.id)
                    ? left.value.id
                    : Number.POSITIVE_INFINITY;
            const rightId =
                typeof right.value.id === 'number' && Number.isFinite(right.value.id)
                    ? right.value.id
                    : Number.POSITIVE_INFINITY;

            if (leftId !== rightId) {
                return leftId - rightId;
            }
            return left.joint.localeCompare(right.joint);
        });

export const formatCalibrationCell = (value: unknown) => {
    if (typeof value === 'number') {
        return Number.isFinite(value) ? String(value) : '-';
    }
    if (typeof value === 'string' && value !== '') {
        return value;
    }
    return '-';
};

const compactTableStyle: CSSProperties = {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '12px',
    lineHeight: '16px',
};
const compactHeaderStyle: CSSProperties = { padding: '4px 8px', fontWeight: 600 };
const compactRowStyle: CSSProperties = { borderTop: '1px solid var(--spectrum-global-color-gray-300)' };
const compactCellStyle: CSSProperties = { padding: '4px 8px' };

export const CalibrationTable = ({
    calibration,
    columns = ['id', 'drive_mode', 'homing_offset', 'range_min', 'range_max'],
    ariaLabel = 'Calibration values',
    className,
}: CalibrationTableProps) => {
    const rows = asCalibrationRows(calibration);
    const isCompact = className === undefined;

    return (
        <table aria-label={ariaLabel} className={className} style={isCompact ? compactTableStyle : undefined}>
            <thead>
                <tr
                    style={
                        isCompact ? { color: 'var(--spectrum-global-color-gray-700)', textAlign: 'left' } : undefined
                    }
                >
                    <th style={isCompact ? compactHeaderStyle : undefined}>Joint</th>
                    {columns.map((column) => (
                        <th key={column} style={isCompact ? compactHeaderStyle : undefined}>
                            {columnLabels[column]}
                        </th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {rows.map((row) => (
                    <tr key={row.joint} style={isCompact ? compactRowStyle : undefined}>
                        <td
                            style={
                                isCompact
                                    ? { ...compactCellStyle, color: 'var(--spectrum-global-color-gray-800)' }
                                    : undefined
                            }
                        >
                            {row.joint}
                        </td>
                        {columns.map((column) => (
                            <td key={column} style={isCompact ? compactCellStyle : undefined}>
                                {formatCalibrationCell(row.value[column])}
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
};
