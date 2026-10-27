import { ReactNode } from 'react';

import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedDataset } from '../../../test-utils/mocks/mock-dataset';
import { getMockedEnvironment, getMockedEnvironmentOutput } from '../../../test-utils/mocks/mock-environment';
import { getMockedRecordingEnvironment } from '../../../test-utils/mocks/mock-recording-environment';
import { render } from '../../../test-utils/render';
import { Index } from './index';

rs.mock('../../../features/robots/runtime-session-provider', () => ({
    RuntimeSessionProvider: ({ children }: { children: ReactNode }) => (
        <div data-testid='runtime-session'>{children}</div>
    ),
    useRuntimeSession: () => ({ state: { episodes_recorded: 0 } }),
}));

rs.mock('./recording-viewer', () => ({ RecordingViewer: () => <div>Recording viewer</div> }));
rs.mock('../../../components/app-footer/app-footer', () => ({ AppFooter: () => null }));

const TestPage = () => {
    const { pathname } = useLocation();
    return pathname.endsWith('/record') ? <Index /> : <div>Dataset page</div>;
};

const renderRecordingPage = () =>
    render(<TestPage />, {
        route: '/projects/project-1/datasets/dataset-1/record',
        path: '/projects/:project_id/datasets/:dataset_id/*',
    });

describe('Recording page environment guard', () => {
    it('automatically associates the sole environment on direct entry', async () => {
        const create = rs.fn();
        server.use(
            http.get('/api/dataset/{dataset_id}/recording-environments', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/environments', () =>
                HttpResponse.json([getMockedEnvironmentOutput()])
            ),
            http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset())),
            http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/environments/{environment_id}', () =>
                HttpResponse.json(getMockedEnvironment())
            ),
            http.post('/api/dataset/{dataset_id}/recording-environments', () => {
                create();
                return HttpResponse.json(getMockedRecordingEnvironment(), { status: 201 });
            })
        );
        renderRecordingPage();

        expect(await screen.findByText('Recording viewer')).toBeInTheDocument();
        expect(create).toHaveBeenCalledOnce();
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('checks direct entry and returns to the dataset page on cancellation', async () => {
        const user = userEvent.setup();
        server.use(
            http.get('/api/dataset/{dataset_id}/recording-environments', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/environments', () =>
                HttpResponse.json([
                    getMockedEnvironmentOutput(),
                    getMockedEnvironmentOutput({ id: 'environment-2', name: 'Second environment' }),
                ])
            )
        );
        renderRecordingPage();

        expect(await screen.findByRole('dialog')).toBeInTheDocument();
        expect(screen.queryByTestId('runtime-session')).not.toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Cancel' }));
        expect(await screen.findByText('Dataset page')).toBeInTheDocument();
    });

    it('waits for environment details without flashing a selection dialog or mounting a runtime session', async () => {
        let resolveEnvironment!: () => void;
        const pendingEnvironment = new Promise<void>((resolve) => {
            resolveEnvironment = resolve;
        });
        const requestedEnvironment = rs.fn();
        server.use(
            http.get('/api/dataset/{dataset_id}/recording-environments', () =>
                HttpResponse.json([getMockedRecordingEnvironment()])
            ),
            http.get('/api/projects/{project_id}/environments', () =>
                HttpResponse.json([getMockedEnvironmentOutput()])
            ),
            http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset())),
            http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/environments/{environment_id}', async () => {
                requestedEnvironment();
                await pendingEnvironment;
                return HttpResponse.json(getMockedEnvironment());
            })
        );
        renderRecordingPage();

        await waitFor(() => expect(requestedEnvironment).toHaveBeenCalledOnce());
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        expect(screen.queryByTestId('runtime-session')).not.toBeInTheDocument();
        resolveEnvironment();
        expect(await screen.findByText('Recording viewer')).toBeInTheDocument();
        expect(screen.getByTestId('runtime-session')).toBeInTheDocument();
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
});
