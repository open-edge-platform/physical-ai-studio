// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaRemoteTrainer, SchemaRemoteTrainerHealth } from '../../api/openapi-spec';

export const getMockedRemoteTrainer = (overrides: Partial<SchemaRemoteTrainer> = {}): SchemaRemoteTrainer => ({
    id: 'trainer-1',
    name: 'managed-trainer',
    connection_mode: 'direct',
    url: 'https://trainer.example.test/api',
    connection: { connection_mode: 'direct', url: 'https://trainer.example.test/api' },
    created_at: '2026-07-14T12:00:00Z',
    ...overrides,
});

export const getMockedRemoteTrainerHealth = (
    overrides: Partial<SchemaRemoteTrainerHealth> = {}
): SchemaRemoteTrainerHealth => ({
    remote_trainer_id: 'trainer-1',
    status: 'healthy',
    checked_at: '2026-07-16T12:00:00Z',
    latency_ms: 24,
    devices: [{ type: 'cuda', name: 'NVIDIA A100', memory: 85899345920, index: 0 }],
    storage: { total_bytes: 1_000_000_000_000, free_bytes: 600_000_000_000 },
    reason_code: null,
    ...overrides,
});
