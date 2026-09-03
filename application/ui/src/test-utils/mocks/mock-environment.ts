// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaEnvironmentWithRelations } from '../../api/openapi-spec';

export const getMockedEnvironment = (
    overrides: Partial<SchemaEnvironmentWithRelations> = {}
): SchemaEnvironmentWithRelations => ({
    id: 'environment-1',
    name: 'lab-environment',
    ...overrides,
});
