// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createRef } from 'react';

import { rs } from '@rstest/core';

import { useRuntimeSession } from '../../features/robots/runtime-session-provider';
import { getMockedDataset } from './mock-dataset';
import { getMockedEnvironment } from './mock-environment';

type RuntimeSession = ReturnType<typeof useRuntimeSession>;
type RuntimeSessionState = RuntimeSession['state'];

export const getMockedRuntimeSessionState = (overrides: Partial<RuntimeSessionState> = {}): RuntimeSessionState => ({
    connected: true,
    follower_source: 'teleop',
    has_leader: true,
    model_loaded: false,
    task: null,
    dataset_loaded: true,
    is_recording: false,
    episodes_recorded: 0,
    ...overrides,
});

export const getMockedRuntimeMutation = (isPending = false) => ({ mutate: rs.fn(), isPending });

export const getMockedRuntimeSession = (
    overrides: Partial<Record<keyof RuntimeSession, unknown>> = {}
): RuntimeSession =>
    ({
        observation: createRef(),
        actions: createRef(),
        environment: getMockedEnvironment(),
        dataset: getMockedDataset(),
        state: getMockedRuntimeSessionState(),
        startEpisode: getMockedRuntimeMutation(),
        saveEpisode: getMockedRuntimeMutation(),
        discardEpisode: getMockedRuntimeMutation(),
        readyForRecording: true,
        isConnected: true,
        ...overrides,
    }) as unknown as RuntimeSession;
