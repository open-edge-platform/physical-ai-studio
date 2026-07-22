import { describe, expect, it } from 'vitest';

import { buildRobotBody } from './form-data';

describe('buildRobotBody', () => {
    it('preserves SO101 calibration when form-owned fields change', () => {
        const body = buildRobotBody(
            {
                name: 'Updated arm',
                payload: {
                    connection_string: '/dev/ttyACM1',
                    serial_number: 'SO101-001',
                    calibration: {
                        shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 10, range_min: -100, range_max: 100 },
                    },
                },
            },
            'SO101_Follower',
            'robot-1'
        );

        expect(body).not.toBeNull();
        expect(body?.payload).toEqual({
            connection_string: '/dev/ttyACM1',
            serial_number: 'SO101-001',
            calibration: {
                shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 10, range_min: -100, range_max: 100 },
            },
        });
    });

    it('preserves unmodeled bimanual payload fields', () => {
        const body = buildRobotBody(
            {
                name: 'Updated bimanual arm',
                payload: {
                    connection_string_left: '192.168.1.2',
                    connection_string_right: '192.168.1.3',
                    serial_number: '',
                },
            },
            'Trossen_Bimanual_WidowXAI_Follower',
            'robot-1'
        );

        expect(body).not.toBeNull();
        expect(body?.payload).toEqual({
            connection_string_left: '192.168.1.2',
            connection_string_right: '192.168.1.3',
            serial_number: '',
        });
    });
});
