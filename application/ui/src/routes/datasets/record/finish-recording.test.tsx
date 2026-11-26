import { toast } from '@geti-ui/ui';
import { rs } from '@rstest/core';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaAckMessage, SchemaSessionState } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedDataset } from '../../../test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { render } from '../../../test-utils/render';
import { Index } from './index';

const session: { state: Partial<SchemaSessionState> } = { state: {} };
const ok: SchemaAckMessage = { type: 'ack', request_id: '', ok: true };
let reply: () => Promise<SchemaAckMessage> = async () => ok;
const send = rs.fn(async () => {
    const ack = await reply();
    // As the backend does: the dataset is unloaded whether or not it was saved.
    session.state = { ...session.state, dataset: null };
    return ack;
});

rs.mock('../../../features/runtime-session/runtime-session-provider', () => ({
    environmentDraft: () => ({}),
    // Mounted while the runtime session is open.
    RuntimeSessionProvider: ({ children }: { children: React.ReactNode }) => (
        <div data-testid='runtime-session'>{children}</div>
    ),
    useRuntimeSession: () => ({ state: session.state, send }),
}));
// The footer and the workspace open their own sockets; neither matters here.
rs.mock('../../../components/app-footer/app-footer', () => ({ AppFooter: () => null }));
rs.mock('../../../features/runtime-session/workspace/session-workspace', () => ({
    SessionWorkspace: () => <div>workspace</div>,
}));

const notSaved: SchemaAckMessage = {
    type: 'ack',
    request_id: '',
    ok: false,
    error: {
        error_code: 'recording_not_saved',
        message:
            "The recording was not saved into dataset 'Blocks': RuntimeError: Copying back failed (No space " +
            'left on device). The recorded episodes are left in /cache/3f2a',
    },
};

/** The record page through its route, as the app renders it. */
const renderPage = async () => {
    render(<Index />, {
        route: '/projects/p/datasets/d/record',
        path: '/projects/:project_id/datasets/:dataset_id/record',
    });
    return screen.findByText('Adding Episode');
};

describe('Finishing a recording', () => {
    beforeEach(() => {
        send.mockClear();
        reply = async () => ok;
        session.state = { session_id: 's', dataset: { dataset_id: 'd' } as SchemaSessionState['dataset'] };
        server.use(
            http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset({ id: 'd' }))),
            http.get('/api/projects/{project_id}/environments/{environment_id}', () =>
                HttpResponse.json(getMockedEnvironment())
            ),
            http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([]))
        );
    });

    it('closes the session and says where the recording was left when it was not saved', async () => {
        reply = async () => notSaved;
        fireEvent.click(await renderPage());

        expect(await screen.findByText(/The recorded episodes are left in \/cache\/3f2a/)).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'The recording was not saved into the dataset' })).toBeVisible();
        // The session is gone, as on main when Finish left the page: no workspace, no controls.
        expect(screen.queryByTestId('runtime-session')).not.toBeInTheDocument();
        expect(screen.queryByText('workspace')).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Leave' })).toBeVisible();
        expect(send).toHaveBeenCalledTimes(1);
    });

    it.each(['connection_closed', 'command_timeout'])(
        'stays neutral when the ack failed in transit (%s)',
        async (error_code) => {
            reply = async () => ({
                type: 'ack',
                request_id: '',
                ok: false,
                error: { error_code, message: 'Connection closed' },
            });
            fireEvent.click(await renderPage());

            expect(
                await screen.findByRole('heading', { name: 'Studio could not tell whether the recording was saved' })
            ).toBeVisible();
            expect(screen.getByRole('button', { name: 'Check the dataset' })).toBeVisible();
            expect(screen.queryByText(/not saved/)).not.toBeInTheDocument();
            expect(screen.queryByTestId('runtime-session')).not.toBeInTheDocument();
        }
    );

    it('leaves for the dataset with a warning when only removing the cache failed', async () => {
        const info = rs.spyOn(toast, 'info');
        reply = async () => ({
            ...ok,
            result: {
                warning: "The recording was saved into dataset 'Blocks', but its cache /cache/x could not be removed",
            },
        });
        fireEvent.click(await renderPage());

        await waitFor(() => expect(screen.queryByText('workspace')).not.toBeInTheDocument());
        expect(info).toHaveBeenCalledWith(expect.stringContaining('could not be removed'));
        expect(screen.queryByText(/not saved into|could not tell/)).not.toBeInTheDocument(); // no recovery view
    });

    it('finishes once, however often Finish is pressed while it saves', async () => {
        let answer: (ack: SchemaAckMessage) => void = () => {};
        reply = () => new Promise((resolve) => (answer = resolve));
        fireEvent.click(await renderPage());
        await waitFor(() => expect(send).toHaveBeenCalledTimes(1));

        fireEvent.click(screen.getByText('Saving recording…'));
        fireEvent.click(screen.getByText('Saving recording…'));
        answer(notSaved);

        expect(await screen.findByRole('button', { name: 'Leave' })).toBeVisible();
        expect(send).toHaveBeenCalledTimes(1);
    });

    it('leaves for the dataset once the recording is saved', async () => {
        fireEvent.click(await renderPage());

        await waitFor(() => expect(screen.queryByText('workspace')).not.toBeInTheDocument());
        expect(send).toHaveBeenCalledWith({ command: 'unload_dataset' });
    });
});
