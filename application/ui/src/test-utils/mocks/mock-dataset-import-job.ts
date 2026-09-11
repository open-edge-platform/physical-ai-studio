// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaDatasetImportJob } from '../../api/openapi-spec';

export const getMockedDatasetImportJob = (overrides: Partial<SchemaDatasetImportJob> = {}): SchemaDatasetImportJob => ({
    id: 'import-job-1',
    project_id: 'project-1',
    type: 'dataset_import',
    status: 'running',
    progress: 10,
    message: 'Job created',
    start_time: null,
    end_time: null,
    created_at: '2026-07-14T09:00:00Z',
    extra_info: {},
    ...overrides,
    payload: {
        step: 'awaiting_archive_upload',
        archive_staging_id: 'staging-1',
        format_hint: 'auto',
        ...(overrides.payload ?? {}),
    },
});
