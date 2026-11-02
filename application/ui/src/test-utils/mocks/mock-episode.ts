// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaEpisodeInfo } from '../../api/openapi-spec';

export const getMockedEpisode = (overrides: Partial<SchemaEpisodeInfo> = {}): SchemaEpisodeInfo => ({
    episode_index: 0,
    tasks: [],
    length: 10,
    fps: 30,
    ...overrides,
});

export const getMockedEpisodes = (
    overridesList: Partial<SchemaEpisodeInfo>[] = [{ episode_index: 0 }, { episode_index: 1 }]
): SchemaEpisodeInfo[] => overridesList.map((overrides) => getMockedEpisode(overrides));
