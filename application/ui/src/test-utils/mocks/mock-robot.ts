// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaUnavailableRobot } from '../../api/openapi-spec';
import { SchemaRobot } from '../../features/robots/robot-types';

type SO101Follower = Extract<SchemaRobot, { type: 'SO101_Follower' }>;
type SO101Leader = Extract<SchemaRobot, { type: 'SO101_Leader' }>;

export const getMockedFollowerRobot = (overrides: Partial<SO101Follower> = {}): SO101Follower => ({
    id: 'follower-1',
    name: 'Follower arm 1',
    type: 'SO101_Follower',
    payload: { connection_string: '', serial_number: 'SO101-001', calibration: {} },
    ...overrides,
});

export const getMockedLeaderRobot = (overrides: Partial<SO101Leader> = {}): SO101Leader => ({
    id: 'leader-1',
    name: 'Leader arm 1',
    type: 'SO101_Leader',
    payload: { connection_string: '', serial_number: 'SO101-002', calibration: {} },
    ...overrides,
});

export const getMockedUnavailableRobot = (overrides: Partial<SchemaUnavailableRobot> = {}): SchemaUnavailableRobot => ({
    id: 'unavailable-1',
    name: 'Missing arm',
    type: 'Acme_Follower',
    payload: {},
    unavailable: true,
    ...overrides,
});
