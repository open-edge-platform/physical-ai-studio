// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaRemoteTrainer } from '../../api/openapi-spec';

export const getMockedRemoteTrainer = (overrides: Partial<SchemaRemoteTrainer> = {}): SchemaRemoteTrainer => ({
    id: 'trainer-1',
    name: 'managed-trainer',
    url: 'https://trainer.example.test/api',
    created_at: '2026-07-14T12:00:00Z',
    ...overrides,
});
