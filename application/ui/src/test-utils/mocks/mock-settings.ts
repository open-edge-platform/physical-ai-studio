import { SchemaRerunSettings, SchemaUserSettingsResponse } from '../../api/openapi-spec';

export const getMockedRerunSettings = (overrides: Partial<SchemaRerunSettings> = {}): SchemaRerunSettings => ({
    enabled: true,
    mode: 'spawn',
    open_viewer: false,
    spawn_port: 9876,
    connect_addr: '127.0.0.1:9876',
    ...overrides,
});

export const getMockedUserSettings = (
    overrides: Partial<SchemaUserSettingsResponse> = {}
): SchemaUserSettingsResponse => ({
    video_recording: { preset: null },
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
    rerun: getMockedRerunSettings(),
    ...overrides,
});
