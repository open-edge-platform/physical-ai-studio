// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaDatasetOutput } from '../../api/openapi-spec';

export const getMockedDataset = (overrides: Partial<SchemaDatasetOutput> = {}): SchemaDatasetOutput => ({
    id: 'dataset-1',
    name: 'pick-dataset',
    default_task: 'manipulation',
    project_id: 'project-1',
    environment_id: 'environment-1',
    ...overrides,
});
