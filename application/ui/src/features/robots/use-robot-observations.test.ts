import { describe, expect, it } from 'vitest';

import { parseRobotObservationMessage } from './use-robot-observations';

describe('parseRobotObservationMessage', () => {
    it('turns an observation frame into joint state', () => {
        expect(
            parseRobotObservationMessage({
                event: 'observation',
                data: { 'shoulder_pan.pos': 12.3, 'gripper.pos': 0 },
            })
        ).toEqual({
            type: 'observation',
            joints: [
                { name: 'shoulder_pan.pos', value: 12.3 },
                { name: 'gripper.pos', value: 0 },
            ],
        });
    });

    it('ignores unknown events', () => {
        expect(parseRobotObservationMessage({ event: 'lifecycle', data: {} })).toEqual({ type: 'ignored' });
        expect(parseRobotObservationMessage(null)).toEqual({ type: 'ignored' });
        expect(parseRobotObservationMessage('observation')).toEqual({ type: 'ignored' });
    });

    it('reads an error frame', () => {
        expect(
            parseRobotObservationMessage({
                event: 'error',
                message: 'Device serial:ttyACM0 is already in use by another session.',
                error_code: 'robot_device_already_owned',
            })
        ).toEqual({
            type: 'error',
            message: 'Device serial:ttyACM0 is already in use by another session.',
            errorCode: 'robot_device_already_owned',
        });
    });

    it('treats a connected state frame as readiness, not joint data', () => {
        expect(parseRobotObservationMessage({ event: 'state', data: { connected: true } })).toEqual({ type: 'state' });
    });
});
