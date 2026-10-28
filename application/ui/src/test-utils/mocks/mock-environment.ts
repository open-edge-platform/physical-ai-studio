// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaEnvironmentOutput, SchemaEnvironmentWithRelations } from '../../api/openapi-spec';

export const getMockedEnvironment = (
    overrides: Partial<SchemaEnvironmentWithRelations> = {}
): SchemaEnvironmentWithRelations => ({
    id: 'environment-1',
    name: 'lab-environment',
    ...overrides,
});

export const getMockedEnvironmentOutput = (
    overrides: Partial<SchemaEnvironmentOutput> = {}
): SchemaEnvironmentOutput => ({
    id: 'environment-1',
    name: 'lab-environment',
    ...overrides,
});
