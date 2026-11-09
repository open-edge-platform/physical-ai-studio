// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaSessionEnvironment } from '../../api/openapi-spec';

export const getMockedSessionEnvironment = (
    overrides: Partial<SchemaSessionEnvironment> = {}
): SchemaSessionEnvironment => ({
    environment_id: 'environment-1',
    name: 'Bench',
    robot_id: 'follower-1',
    leader_id: 'leader-1',
    joint_names: [],
    cameras: [],
    ...overrides,
});
