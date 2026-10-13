// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { RobotSimulation, useRobotSimulation } from './use-robot-simulation';

const simulation = (running: boolean, started_by: 'studio' | 'external'): RobotSimulation => ({
    running,
    started_by: running ? started_by : null,
    owner_name: 'sim',
    http_url: running ? 'http://127.0.0.1:8080' : null,
    viewer_url: running ? 'http://127.0.0.1:8081' : null,
    scene: null,
    cameras: [],
});

const renderWithResponses = (responses: RobotSimulation[]) => {
    let call = 0;
    server.use(
        http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
            HttpResponse.json(responses[Math.min(call++, responses.length - 1)])
        )
    );
    const queryClient = createQueryClient();

    return renderHook(() => useRobotSimulation('project-1', 'robot-1', false), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
    });
};

const refetch = async (result: { current: ReturnType<typeof useRobotSimulation> }) => {
    await act(async () => {
        await result.current.refetch();
    });
    // Let each probe commit on its own, as the 2 s polling does.
    await act(() => new Promise((resolve) => setTimeout(resolve, 20)));
};

describe('useRobotSimulation', () => {
    it('keeps an outside simulation running through one missed probe', async () => {
        const { result } = renderWithResponses([
            simulation(true, 'external'),
            simulation(false, 'external'),
            simulation(false, 'external'),
        ]);
        await refetch(result);
        await waitFor(() => expect(result.current.data?.running).toBe(true));

        await refetch(result);
        expect(result.current.data?.running).toBe(true);

        await refetch(result);
        await waitFor(() => expect(result.current.data?.running).toBe(false));
    });

    it("follows Studio's own simulation at once", async () => {
        const { result } = renderWithResponses([simulation(true, 'studio'), simulation(false, 'studio')]);
        await refetch(result);
        await waitFor(() => expect(result.current.data?.running).toBe(true));

        await refetch(result);
        await waitFor(() => expect(result.current.data?.running).toBe(false));
    });
});
