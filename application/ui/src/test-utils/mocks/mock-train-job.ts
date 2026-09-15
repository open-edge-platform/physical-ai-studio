// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaTrainJob } from '../../api/openapi-spec';
import { getMockedTrainJobPayload } from './mock-train-job-payload';

export const getMockedTrainJob = (overrides: Partial<SchemaTrainJob> = {}): SchemaTrainJob =>
    ({
        id: 'job-1',
        project_id: 'project-1',
        type: 'training',
        status: 'running',
        progress: 42,
        message: 'Training',
        start_time: '2026-07-14T10:00:00Z',
        end_time: null,
        created_at: '2026-07-14T09:00:00Z',
        extra_info: {},
        ...overrides,
        payload: getMockedTrainJobPayload(overrides.payload as never),
    }) as SchemaTrainJob;
