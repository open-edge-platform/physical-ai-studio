// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaProjectOutput } from '../../api/openapi-spec';

export const getMockedProject = (overrides: Partial<SchemaProjectOutput> = {}): SchemaProjectOutput => ({
    id: 'project-1',
    name: 'Project',
    datasets: [],
    ...overrides,
});
