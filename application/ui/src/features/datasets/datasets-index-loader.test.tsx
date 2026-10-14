// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { createMemoryRouter } from 'react-router';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { getMockedDataset } from '../../test-utils/mocks/mock-dataset';
import { getMockedProject } from '../../test-utils/mocks/mock-project';
import { datasetsIndexLoader } from './datasets-index-loader';

const PROJECT_ID = 'project-1';
const PROJECT_PATH = '/api/projects/{project_id}';
const FIRST_DATASET_ID = 'dataset-1';
const SECOND_DATASET_ID = 'dataset-2';

const getProject = () =>
    getMockedProject({
        id: PROJECT_ID,
        datasets: [
            getMockedDataset({ id: FIRST_DATASET_ID, project_id: PROJECT_ID }),
            getMockedDataset({ id: SECOND_DATASET_ID, project_id: PROJECT_ID }),
        ],
    });

const createRouter = (initialEntry = `/projects/${PROJECT_ID}/datasets`) =>
    createMemoryRouter(
        [
            { path: '/projects' },
            { path: '/datasets', loader: datasetsIndexLoader },
            {
                path: '/projects/:project_id/datasets',
                children: [{ index: true, loader: datasetsIndexLoader }, { path: ':dataset_id' }],
            },
        ],
        { initialEntries: [initialEntry] }
    );

describe('datasetsIndexLoader', () => {
    beforeEach(() => {
        server.use(http.get(PROJECT_PATH, () => HttpResponse.json(getProject())));
    });

    afterEach(() => {
        localStorage.clear();
    });

    it('redirects to projects when project id is missing', async () => {
        const router = createRouter('/datasets');

        await waitFor(() => expect(router.state.location.pathname).toBe('/projects'));

        router.dispose();
    });

    it('redirects to the first dataset when no dataset was previously used', async () => {
        const router = createRouter();

        await waitFor(() =>
            expect(router.state.location.pathname).toBe(`/projects/${PROJECT_ID}/datasets/${FIRST_DATASET_ID}`)
        );

        router.dispose();
    });

    it('redirects to the previously used dataset when it belongs to the project', async () => {
        localStorage.setItem(`physicalai:lastDataset:${PROJECT_ID}`, SECOND_DATASET_ID);
        const router = createRouter();

        await waitFor(() =>
            expect(router.state.location.pathname).toBe(`/projects/${PROJECT_ID}/datasets/${SECOND_DATASET_ID}`)
        );

        router.dispose();
    });

    it('clears a stale dataset id and redirects to the first dataset', async () => {
        localStorage.setItem(`physicalai:lastDataset:${PROJECT_ID}`, 'stale-dataset');
        const router = createRouter();

        await waitFor(() =>
            expect(router.state.location.pathname).toBe(`/projects/${PROJECT_ID}/datasets/${FIRST_DATASET_ID}`)
        );

        expect(localStorage.getItem(`physicalai:lastDataset:${PROJECT_ID}`)).toBeNull();
        router.dispose();
    });

    it('keeps the datasets index selected when project has no datasets', async () => {
        server.use(http.get(PROJECT_PATH, () => HttpResponse.json(getMockedProject({ id: PROJECT_ID }))));
        const router = createRouter();

        await waitFor(() => expect(router.state.navigation.state).toBe('idle'));

        expect(router.state.location.pathname).toBe(`/projects/${PROJECT_ID}/datasets`);
        router.dispose();
    });

    it('keeps the datasets index selected when project request fails', async () => {
        server.use(http.get(PROJECT_PATH, () => HttpResponse.json({ detail: [] }, { status: 500 })));
        const router = createRouter();

        await waitFor(() => expect(router.state.navigation.state).toBe('idle'));

        expect(router.state.location.pathname).toBe(`/projects/${PROJECT_ID}/datasets`);
        router.dispose();
    });
});
