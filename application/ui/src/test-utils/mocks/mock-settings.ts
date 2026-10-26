// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaUserSettingsResponse } from '../../api/openapi-spec';

export const getMockedSettings = (overrides: Partial<SchemaUserSettingsResponse> = {}): SchemaUserSettingsResponse => ({
    trainer: {
        request_timeout_s: 30,
        download_read_timeout_s: 120,
        stream_reconnect_max_s: 900,
        stream_reconnect_backoff_max_s: 30,
    },
    huggingface: { hf_token: null },
    ssh: {
        connect_timeout_s: 10,
        command_timeout_s: 15,
        preflight_timeout_s: 30,
        image_pull_timeout_s: 1800,
        trainer_shm_size_gb: 32,
    },
    hotkeys: { bindings: {} },
    ...overrides,
});
