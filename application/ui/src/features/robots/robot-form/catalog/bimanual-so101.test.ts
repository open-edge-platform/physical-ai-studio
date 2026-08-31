import { describe, expect, it } from 'vitest';

import { buildBimanualSO101Body } from './bimanual-so101';

const formData = {
    name: 'Dual arm',
    payload: {
        left_serial_number: 'left-arm',
        right_serial_number: 'right-arm',
        left_calibration: {
            shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 0, range_min: 0, range_max: 4095 },
        },
        right_calibration: {
            shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 0, range_min: 0, range_max: 4095 },
        },
        baudrate: 1000000,
        role: 'follower' as const,
        disable_torque_on_disconnect: true,
    },
};

describe('buildBimanualSO101Body', () => {
    it('builds a follower payload from two selected SO101 robots', () => {
        expect(
            buildBimanualSO101Body(formData, 'BimanualSO101_Follower', '00000000-0000-0000-0000-000000000001')
        ).toMatchObject({
            type: 'BimanualSO101_Follower',
            payload: {
                left_serial_number: formData.payload.left_serial_number,
                right_serial_number: formData.payload.right_serial_number,
                left_calibration: formData.payload.left_calibration,
                right_calibration: formData.payload.right_calibration,
                role: 'follower',
            },
        });
    });

    it('requires selections for both arms', () => {
        expect(
            buildBimanualSO101Body(
                { ...formData, payload: { ...formData.payload, right_calibration: null } },
                'BimanualSO101_Follower',
                '00000000-0000-0000-0000-000000000001'
            )
        ).toBeNull();
    });
});
