// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { PartControlsProvider, usePartControls } from './part-controls';

const send = rs.fn(async () => ({
    type: 'ack',
    request_id: '1',
    ok: false,
    error: {
        message: 'This would unload dataset',
        error_code: 'incompatible_parts',
        parts: [{ part: 'dataset', name: 'Blocks', reasons: ["camera 'wrist': no such camera"] }],
    },
}));

rs.mock('../runtime-session-provider', () => ({
    savedEnvironmentDraft: rs.fn(),
    useRuntimeSession: () => ({ state: null, loading: null, send }),
}));
rs.mock('../../projects/use-project', () => ({
    useProjectId: () => ({ project_id: 'project-1' }),
    useProject: () => ({ datasets: [] }),
}));

const draft = { id: 'environment-2', name: 'Franka bench', robots: [], cameras: [] };

const LoadButton = () => {
    const { loadEnvironmentDraft } = usePartControls();
    return <button onClick={() => loadEnvironmentDraft(draft)}>Load it</button>;
};

describe('PartControlsProvider loading an environment draft', () => {
    it('asks before unloading parts the environment does not fit, as a swap does', async () => {
        server.use(
            http.get('/api/projects/{project_id}/environments', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/models', () => HttpResponse.json([]))
        );
        render(
            <PartControlsProvider>
                <LoadButton />
            </PartControlsProvider>
        );

        await userEvent.setup().click(screen.getByRole('button', { name: 'Load it' }));

        expect(await screen.findByText('Unload parts that no longer fit?')).toBeInTheDocument();
        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'load_environment',
                environment: draft,
                unload_incompatible: false,
            })
        );
    });
});
