// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it, vi } from 'vitest';

import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { getMockedDataset } from '../../test-utils/mocks/mock-dataset';
import { render } from '../../test-utils/render';
import { RenameDatasetDialog } from './rename-dataset-dialog';

const DATASET_ID = 'dataset-1';
const PROJECT_ID = 'project-1';

const datasetQueryKey = ['get', '/api/dataset/{dataset_id}', { params: { path: { dataset_id: DATASET_ID } } }] as const;
const projectQueryKey = [
    'get',
    '/api/projects/{project_id}',
    { params: { path: { project_id: PROJECT_ID } } },
] as const;

const dataset = getMockedDataset({ id: DATASET_ID, project_id: PROJECT_ID, name: 'pick-dataset' });

const renderDialog = (
    onDone: (dataset: SchemaDatasetOutput | undefined) => void,
    queryClient: ReturnType<typeof createQueryClient>
) => render(<RenameDatasetDialog dataset={dataset} onDone={onDone} />, { queryClient });

const renameViaUi = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
    const textField = await screen.findByLabelText(/Dataset name/);
    await user.clear(textField);
    await user.type(textField, name);
    await user.click(screen.getByRole('button', { name: 'Save' }));
};

describe('RenameDatasetDialog', () => {
    it('submits trimmed name and calls onDone with updated dataset', async () => {
        let receivedBody: unknown;
        let receivedDatasetId: string | undefined;
        const updatedDataset = getMockedDataset({ id: DATASET_ID, project_id: PROJECT_ID, name: 'renamed' });

        server.use(
            http.put('/api/dataset/{dataset_id}', async ({ request, params }) => {
                receivedBody = await request.json();
                receivedDatasetId = params.dataset_id;
                return HttpResponse.json(updatedDataset);
            })
        );

        const user = userEvent.setup();
        const onDone = vi.fn();
        renderDialog(onDone, createQueryClient());

        await renameViaUi(user, '  renamed  ');

        await waitFor(() => expect(onDone).toHaveBeenCalledWith(updatedDataset));

        expect(receivedBody).toEqual({ name: 'renamed' });
        expect(receivedDatasetId).toBe(DATASET_ID);
    });

    it('invalidates dataset and project queries after rename', async () => {
        server.use(
            http.put('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset({ name: 'renamed' })))
        );

        const queryClient = createQueryClient();
        queryClient.setQueryData(datasetQueryKey, dataset);
        queryClient.setQueryData(projectQueryKey, { id: PROJECT_ID, name: 'test-project' });

        const user = userEvent.setup();
        renderDialog(vi.fn(), queryClient);

        await renameViaUi(user, 'renamed');

        await waitFor(() => expect(queryClient.getQueryState(datasetQueryKey)?.isInvalidated).toBe(true));
        expect(queryClient.getQueryState(projectQueryKey)?.isInvalidated).toBe(true);
    });

    it('calls onDone(undefined) on cancel without submitting', async () => {
        let putCalled = false;
        server.use(
            http.put('/api/dataset/{dataset_id}', () => {
                putCalled = true;
                return HttpResponse.json(dataset);
            })
        );

        const user = userEvent.setup();
        const onDone = vi.fn();
        renderDialog(onDone, createQueryClient());

        await user.click(await screen.findByRole('button', { name: 'Cancel' }));

        expect(onDone).toHaveBeenCalledWith(undefined);
        expect(putCalled).toBe(false);
    });

    it('field prefilled and save disabled when name is whitespace only', async () => {
        const user = userEvent.setup();
        renderDialog(vi.fn(), createQueryClient());

        const textField = await screen.findByLabelText(/Dataset name/);
        expect(textField).toHaveValue(dataset.name);

        await user.clear(textField);

        expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
    });
});
