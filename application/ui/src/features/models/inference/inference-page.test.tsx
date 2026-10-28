import { ReactNode } from 'react';

import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaEnvironmentWithRelations } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedEnvironment, getMockedEnvironmentOutput } from '../../../test-utils/mocks/mock-environment';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { getMockedRecordingEnvironment } from '../../../test-utils/mocks/mock-recording-environment';
import { render } from '../../../test-utils/render';
import { InferencePage } from './inference-page';

rs.mock('../../runtime-session/runtime-session-provider', () => ({
    environmentDraft: (environment: SchemaEnvironmentWithRelations) => environment,
    RuntimeSessionProvider: ({
        environment,
        children,
    }: {
        environment: SchemaEnvironmentWithRelations;
        children: ReactNode;
    }) => (
        <div data-testid='runtime-session' data-environment-id={environment.id}>
            {children}
        </div>
    ),
}));
rs.mock('../../runtime-session/workspace/session-workspace', () => ({
    SessionWorkspace: () => <div>Inference viewer</div>,
}));
rs.mock('../../../components/app-footer/app-footer', () => ({ AppFooter: () => null }));

const renderPage = () =>
    render(<InferencePage />, {
        route: '/projects/project-1/models/model-1/inference/openvino',
        path: '/projects/:project_id/models/:model_id/inference/:backend',
    });

describe('InferencePage environment selection', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/models/{model_id}', () =>
                HttpResponse.json({
                    model: getMockedModel(),
                    exports: [],
                    training_summary: null,
                    hparams: null,
                })
            ),
            http.get('/api/models/{model_id}/tasks', () => HttpResponse.json([])),
            http.get('/api/system/devices/inference', () => HttpResponse.json([])),
            http.get('/api/dataset/{dataset_id}/recording-environments', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/environments', () =>
                HttpResponse.json([
                    getMockedEnvironmentOutput(),
                    getMockedEnvironmentOutput({ id: 'environment-2', name: 'Recorded environment' }),
                ])
            ),
            http.get('/api/projects/{project_id}/environments/{environment_id}', ({ params }) =>
                HttpResponse.json(getMockedEnvironment({ id: params.environment_id }))
            )
        );
    });

    it('uses the recording association rather than the first project environment', async () => {
        server.use(
            http.get('/api/dataset/{dataset_id}/recording-environments', () =>
                HttpResponse.json([getMockedRecordingEnvironment({ environment_id: 'environment-2' })])
            )
        );
        renderPage();

        expect(await screen.findByTestId('runtime-session')).toHaveAttribute('data-environment-id', 'environment-2');
    });

    it('does not open a hardware session until an unassociated environment is explicitly selected', async () => {
        const user = userEvent.setup();
        renderPage();

        expect(await screen.findByText('Select an inference environment')).toBeInTheDocument();
        expect(screen.queryByTestId('runtime-session')).not.toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: /Select.* Environment/ }));
        await user.click(await screen.findByRole('option', { name: 'Recorded environment' }));

        expect(await screen.findByTestId('runtime-session')).toHaveAttribute('data-environment-id', 'environment-2');
    });

    it('requires explicit selection even when an unassociated project has only one environment', async () => {
        server.use(
            http.get('/api/projects/{project_id}/environments', () => HttpResponse.json([getMockedEnvironmentOutput()]))
        );
        renderPage();

        expect(await screen.findByText('Select an inference environment')).toBeInTheDocument();
        expect(screen.queryByTestId('runtime-session')).not.toBeInTheDocument();
    });

    it('does not open a session when no environments exist', async () => {
        server.use(http.get('/api/projects/{project_id}/environments', () => HttpResponse.json([])));
        renderPage();

        expect(await screen.findByText('Create an environment before starting inference.')).toBeInTheDocument();
        expect(screen.queryByTestId('runtime-session')).not.toBeInTheDocument();
    });
});
