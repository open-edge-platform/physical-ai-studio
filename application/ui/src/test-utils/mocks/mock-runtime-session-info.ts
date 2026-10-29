// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaRuntimeSessionInfo } from '../../api/openapi-spec';

export const getMockedRuntimeSessionInfo = (
    overrides: Partial<SchemaRuntimeSessionInfo> = {}
): SchemaRuntimeSessionInfo => ({
    session_name: 'rt-follower-1',
    follower_id: 'follower-1',
    status: 'running',
    ...overrides,
});
