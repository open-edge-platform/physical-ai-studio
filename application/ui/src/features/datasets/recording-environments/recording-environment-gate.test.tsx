import { StrictMode } from 'react';

import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaEnvironmentOutput, SchemaRecordingEnvironment } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedEnvironmentOutput } from '../../../test-utils/mocks/mock-environment';
import { getMockedRecordingEnvironment } from '../../../test-utils/mocks/mock-recording-environment';
import { render } from '../../../test-utils/render';
import { RecordingEnvironmentGate } from './recording-environment-gate';

const firstEnvironment = getMockedEnvironmentOutput();
const secondEnvironment = getMockedEnvironmentOutput({ id: 'environment-2', name: 'Second environment' });
const association = getMockedRecordingEnvironment();

const mockEnvironments = (environments: SchemaEnvironmentOutput[], associations: SchemaRecordingEnvironment[] = []) => {
    const create = rs.fn<(body: unknown) => void>();
    server.use(
        http.get('/api/projects/{project_id}/environments', () => HttpResponse.json(environments)),
        http.get('/api/dataset/{dataset_id}/recording-environments', () => HttpResponse.json(associations)),
        http.post('/api/dataset/{dataset_id}/recording-environments', async ({ request }) => {
            create(await request.json());
            return HttpResponse.json(association, { status: 201 });
        })
    );
    return create;
};

const renderGate = () => {
    const onCancel = rs.fn();
    render(
        <StrictMode>
            <RecordingEnvironmentGate project_id='project-1' dataset_id='dataset-1' onCancel={onCancel}>
                {(environment) => <div>Ready: {environment.id}</div>}
            </RecordingEnvironmentGate>
        </StrictMode>
    );
    return onCancel;
};

describe('RecordingEnvironmentGate', () => {
    it('reuses an existing association without prompting or creating another', async () => {
        const create = mockEnvironments([firstEnvironment, secondEnvironment], [association]);
        renderGate();

        expect(await screen.findByText(`Ready: ${association.id}`)).toBeInTheDocument();
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        expect(create).not.toHaveBeenCalled();
    });

    it('automatically associates the sole project environment exactly once in StrictMode', async () => {
        const create = mockEnvironments([firstEnvironment]);
        renderGate();

        expect(await screen.findByText(`Ready: ${association.id}`)).toBeInTheDocument();
        expect(create).toHaveBeenCalledTimes(1);
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        expect(create).toHaveBeenCalledWith({ environment_id: firstEnvironment.id, camera_mappings: [] });
    });

    it('waits for confirmation before associating a selected environment', async () => {
        const user = userEvent.setup();
        const create = mockEnvironments([firstEnvironment, secondEnvironment]);
        renderGate();

        const confirm = await screen.findByRole('button', { name: 'Use environment' });
        expect(confirm).toBeDisabled();
        await user.click(screen.getByRole('button', { name: /Select.* Environment/ }));
        await user.click(await screen.findByRole('option', { name: firstEnvironment.name }));
        expect(create).not.toHaveBeenCalled();
        await user.click(confirm);

        expect(await screen.findByText(`Ready: ${association.id}`)).toBeInTheDocument();
        expect(create).toHaveBeenCalledTimes(1);
    });

    it('allows cancelling without creating an association', async () => {
        const user = userEvent.setup();
        const create = mockEnvironments([firstEnvironment, secondEnvironment]);
        const onCancel = renderGate();
        await user.click(await screen.findByRole('button', { name: 'Cancel' }));

        expect(onCancel).toHaveBeenCalledOnce();
        expect(create).not.toHaveBeenCalled();
    });

    it('keeps the selection open when association fails and allows retrying', async () => {
        const user = userEvent.setup();
        mockEnvironments([firstEnvironment, secondEnvironment]);
        let attempts = 0;
        server.use(
            http.post('/api/dataset/{dataset_id}/recording-environments', () => {
                attempts += 1;
                if (attempts === 1) return HttpResponse.json({ detail: [] }, { status: 500 });
                return HttpResponse.json(association, { status: 201 });
            })
        );
        renderGate();

        await user.click(await screen.findByRole('button', { name: /Select.* Environment/ }));
        await user.click(await screen.findByRole('option', { name: firstEnvironment.name }));
        await user.click(screen.getByRole('button', { name: 'Use environment' }));
        expect(await screen.findByText('Could not associate this environment. Please try again.')).toBeInTheDocument();
        expect(screen.queryByText(`Ready: ${association.id}`)).not.toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Use environment' }));
        expect(await screen.findByText(`Ready: ${association.id}`)).toBeInTheDocument();
        expect(attempts).toBe(2);
    });

    it('offers environment creation when the project has no environments', async () => {
        const create = mockEnvironments([]);
        renderGate();

        expect(await screen.findByRole('button', { name: 'Create environment' })).toHaveAttribute(
            'href',
            '/projects/project-1/environments/new'
        );
        expect(screen.queryByRole('button', { name: 'Use environment' })).not.toBeInTheDocument();
        expect(create).not.toHaveBeenCalled();
    });

    it('allows retrying a failed automatic association without an automatic retry loop', async () => {
        const user = userEvent.setup();
        mockEnvironments([firstEnvironment]);
        const create = rs
            .fn()
            .mockImplementationOnce(() => HttpResponse.json({ detail: 'Failed' }, { status: 500 }))
            .mockImplementationOnce(() => HttpResponse.json(association, { status: 201 }));
        server.use(http.post('/api/dataset/{dataset_id}/recording-environments', create));
        renderGate();

        await user.click(await screen.findByRole('button', { name: 'Try again' }));
        expect(await screen.findByText(`Ready: ${association.id}`)).toBeInTheDocument();
        await waitFor(() => expect(create).toHaveBeenCalledTimes(2));
    });
});
