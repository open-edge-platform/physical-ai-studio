import { describe, expect, it } from 'vitest';

import { isRecoverableRobotControlError, runtimeSocketUrl } from './use-joint-state';

describe('isRecoverableRobotControlError', () => {
    it('classifies a leader connection loss as recoverable', () => {
        expect(isRecoverableRobotControlError('leader_connection_lost')).toBe(true);
    });

    it('keeps session connection failures fatal', () => {
        expect(isRecoverableRobotControlError('robot_connection_failed')).toBe(false);
        expect(isRecoverableRobotControlError(undefined)).toBe(false);
    });
});

describe('runtimeSocketUrl', () => {
    it('uses a distinct URL per follower so share:true does not collide', () => {
        const left = runtimeSocketUrl('project-1', 'follower-left');
        const right = runtimeSocketUrl('project-1', 'follower-right');
        expect(left).toContain('follower-left');
        expect(right).toContain('follower-right');
        expect(left).not.toBe(right);
    });
});
