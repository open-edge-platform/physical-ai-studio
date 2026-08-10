import { describe, expect, it } from 'vitest';

import { isRecoverableRobotControlError } from './use-joint-state';

describe('isRecoverableRobotControlError', () => {
    it('classifies a leader connection loss as recoverable', () => {
        expect(isRecoverableRobotControlError('leader_connection_lost')).toBe(true);
    });

    it('keeps session connection failures fatal', () => {
        expect(isRecoverableRobotControlError('robot_connection_failed')).toBe(false);
        expect(isRecoverableRobotControlError(undefined)).toBe(false);
    });
});
