// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaEpisode, SchemaEpisodeVideo } from '../../api/openapi-spec';

export const getMockedEpisodeVideo = (overrides: Partial<SchemaEpisodeVideo> = {}): SchemaEpisodeVideo => ({
    start: 0,
    end: 3,
    path: 'video.mp4',
    ...overrides,
});

export const getMockedEpisodeDetail = (overrides: Partial<SchemaEpisode> = {}): SchemaEpisode => ({
    episode_index: 0,
    length: 100,
    fps: 30,
    tasks: [],
    actions: [],
    action_keys: [],
    videos: {
        'observation.images.top': getMockedEpisodeVideo({ path: 'top.mp4' }),
        'observation.images.wrist': getMockedEpisodeVideo({ path: 'wrist.mp4' }),
    },
    ...overrides,
});
