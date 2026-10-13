// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it } from 'vitest';

import { rememberedScene, rememberScene, RobotSimulationLaunch, robotSimulationStatus } from './robot-simulation';
import { Simulation } from './use-simulation';

const launch: RobotSimulationLaunch = {
    scenes: [
        { id: 'single_pick_place', display_name: 'Single Pick & Place', description: '' },
        { id: 'yahtzee', display_name: 'Yahtzee', description: '' },
    ],
    default_scene: 'single_pick_place',
    max_seed: 100,
    labels: ['1 arm'],
};

const simulation = (overrides: Partial<Simulation>): Simulation => ({
    status: 'idle',
    cameras: [],
    ...overrides,
});

describe('robotSimulationStatus', () => {
    it.each([
        ['running', 'studio'],
        ['starting', 'starting'],
        ['stopping', 'stopping'],
    ] as const)("is %s → %s for the robot Studio's simulation belongs to", (status, expected) => {
        expect(robotSimulationStatus('robot-a', simulation({ status, robot_id: 'robot-a' }), 'online')).toBe(expected);
    });

    it('is external when the robot is online but Studio runs no simulation for it', () => {
        expect(robotSimulationStatus('robot-a', simulation({ status: 'running', robot_id: 'robot-b' }), 'online')).toBe(
            'external'
        );
        expect(robotSimulationStatus('robot-a', simulation({}), 'online')).toBe('external');
    });

    it('follows the connection status otherwise, also after a failed Studio simulation', () => {
        expect(robotSimulationStatus('robot-a', simulation({ status: 'failed', robot_id: 'robot-a' }), 'offline')).toBe(
            'offline'
        );
        expect(robotSimulationStatus('robot-a', undefined, 'unknown')).toBe('unknown');
    });
});

describe('rememberedScene', () => {
    afterEach(() => window.localStorage.clear());

    it('defaults to the launch default scene', () => {
        expect(rememberedScene('robot-a', launch)).toBe('single_pick_place');
    });

    it('returns the scene last started for the robot', () => {
        rememberScene('robot-a', 'yahtzee');

        expect(rememberedScene('robot-a', launch)).toBe('yahtzee');
        expect(rememberedScene('robot-b', launch)).toBe('single_pick_place');
    });

    it('ignores a remembered scene the robot type no longer offers', () => {
        rememberScene('robot-a', 'garment_fold');

        expect(rememberedScene('robot-a', launch)).toBe('single_pick_place');
    });
});
