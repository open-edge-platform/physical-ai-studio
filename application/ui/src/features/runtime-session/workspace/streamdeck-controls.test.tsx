import { useState } from 'react';

import { TextArea } from '@geti-ui/ui';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaStreamDeckStatus } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedModelControlConfig } from '../../../test-utils/mocks/mock-model-control-config';
import { getMockedStreamDeckStatus } from '../../../test-utils/mocks/mock-streamdeck-status';
import { getMockedStreamDeckTarget } from '../../../test-utils/mocks/mock-streamdeck-target';
import { getMockedStreamDeckTranscript } from '../../../test-utils/mocks/mock-streamdeck-transcript';
import { render } from '../../../test-utils/render';
import { StreamDeckControls } from './streamdeck-controls';

let status: SchemaStreamDeckStatus;

beforeEach(() => {
    status = getMockedStreamDeckStatus();
    server.use(http.get('/api/streamdeck', () => HttpResponse.json(status)));
});

describe('StreamDeckControls', () => {
    it('explains host-side setup and disables Connect without a helper', async () => {
        status.helper_online = false;
        render(<StreamDeckControls sessionId='one' task='Pick block' canConnect />);
        expect(await screen.findByText(/Open the installed host launcher/)).toBeVisible();
        expect(screen.getByText('uv run application/streamdeck/host_launcher.py --install')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Start Stream Deck helper' })).toHaveAttribute(
            'href',
            'physicalai-streamdeck://start'
        );
        expect(screen.getByRole('button', { name: 'Connect Stream Deck' })).toBeDisabled();
    });

    it('starts a native helper with the launch header, then enables Connect after its heartbeat', async () => {
        status.helper_online = false;
        status.can_start_native = true;
        let launches = 0;
        server.use(
            http.post('/api/streamdeck/start', ({ request }) => {
                expect(request.headers.get('x-streamdeck-launch')).toBe('1');
                launches += 1;
                status.helper_online = true;
                return HttpResponse.json(status);
            })
        );
        render(<StreamDeckControls sessionId='one' task='Pick block' canConnect />);
        await userEvent.setup().click(await screen.findByRole('button', { name: 'Start Stream Deck helper' }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Connect Stream Deck' })).toBeEnabled());
        expect(launches).toBe(1);
        expect(screen.queryByRole('button', { name: 'Start Stream Deck helper' })).not.toBeInTheDocument();
    });

    it('connects using the current workspace and task, then disconnects without recording commands', async () => {
        const user = userEvent.setup();
        const updates: string[] = [];
        server.use(
            http.put('/api/streamdeck/sessions/{session_id}', async ({ params, request }) => {
                const body = await request.json();
                updates.push(body.task);
                expect(params.session_id).toBe('one');
                status.target = getMockedStreamDeckTarget({ task: body.task });
                status.connected = true;
                status.device = 'Mini';
                return HttpResponse.json(status);
            }),
            http.delete('/api/streamdeck/sessions/{session_id}', () => {
                status.target = null;
                status.connected = false;
                return HttpResponse.json(status);
            })
        );
        render(<StreamDeckControls sessionId='one' task='Pick block' canConnect />);
        const connect = await screen.findByRole('button', { name: 'Connect Stream Deck' });
        await waitFor(() => expect(connect).toBeEnabled());
        await user.click(connect);
        const disconnect = await screen.findByRole('button', { name: 'Disconnect Stream Deck' });
        await waitFor(() => expect(disconnect).toBeEnabled());
        expect(updates).toEqual(['Pick block']);
        await user.click(disconnect);
        expect(await screen.findByRole('button', { name: 'Connect Stream Deck' })).toBeVisible();
        expect(status.target).toBeNull();
    });

    it('does not take the deck from another workspace', async () => {
        status.target = getMockedStreamDeckTarget({
            session_id: 'other',
            task: 'Other task',
            connection_id: 'other-connection',
        });
        render(<StreamDeckControls sessionId='one' task='Pick block' canConnect />);
        expect(await screen.findByText(/Connected in another workspace/)).toBeVisible();
        expect(screen.getByRole('button', { name: 'Connect Stream Deck' })).toBeDisabled();
    });

    it('reports a failed task synchronization instead of retrying it on every poll', async () => {
        status.target = getMockedStreamDeckTarget();
        let attempts = 0;
        let polls = 0;
        server.use(
            http.get('/api/streamdeck', () => {
                polls += 1;
                return HttpResponse.json(status);
            }),
            http.put('/api/streamdeck/sessions/{session_id}', () => {
                attempts += 1;
                return HttpResponse.error();
            }),
            http.delete('/api/streamdeck/sessions/{session_id}', () => {
                status.target = null;
                return HttpResponse.json(status);
            })
        );
        render(<StreamDeckControls sessionId='one' task='New task' canConnect />);
        expect(await screen.findByText(/Task sync failed/)).toBeVisible();
        await waitFor(() => expect(polls).toBeGreaterThan(1), { timeout: 2000 });
        expect(attempts).toBe(1);
        await userEvent.setup().click(screen.getByRole('button', { name: 'Disconnect Stream Deck' }));
        expect(await screen.findByText(/Helper ready/)).toBeVisible();
        expect(screen.queryByText(/Task sync failed/)).not.toBeInTheDocument();
    });

    it.each(['New task', ''])(
        'synchronizes the attached workspace task, including an empty draft: %s',
        async (task) => {
            status.target = getMockedStreamDeckTarget();
            status.connected = true;
            status.device = 'Mini';
            const updates: string[] = [];
            server.use(
                http.put('/api/streamdeck/sessions/{session_id}', async ({ request }) => {
                    const body = await request.json();
                    updates.push(body.task);
                    status.target = getMockedStreamDeckTarget({ task: body.task });
                    return HttpResponse.json(status);
                })
            );
            render(<StreamDeckControls sessionId='one' task={task} canConnect />);
            await waitFor(() => expect(updates).toEqual([task]));
        }
    );

    it('switches pages only after an explicit click, preserving the inference settings', async () => {
        status.target = getMockedStreamDeckTarget();
        const inference = getMockedModelControlConfig({ hz: 60 });
        let updates = 0;
        server.use(
            http.put('/api/streamdeck/sessions/{session_id}', async ({ request }) => {
                const body = await request.json();
                expect(body.page).toBe('inference');
                expect(body.inference?.hz).toBe(60);
                updates += 1;
                status.target = getMockedStreamDeckTarget({ ...body, connection_id: 'inference-page' });
                return HttpResponse.json(status);
            })
        );
        const view = render(
            <StreamDeckControls sessionId='one' task='Old task' page='inference' inference={inference} canConnect />
        );
        const switchPage = await screen.findByRole('button', { name: 'Use inference page' });
        expect(updates).toBe(0);
        await userEvent.setup().click(switchPage);
        expect(await screen.findByRole('button', { name: 'Disconnect Stream Deck' })).toBeVisible();
        expect(updates).toBe(1);
        view.unmount();
        render(<StreamDeckControls sessionId='one' task='Old task' canConnect />);
        expect(await screen.findByRole('button', { name: 'Use recording page' })).toBeVisible();
        expect(updates).toBe(1);
    });

    it('allows an empty inference draft so the microphone can provide the first task', async () => {
        server.use(
            http.put('/api/streamdeck/sessions/{session_id}', async ({ request }) => {
                const body = await request.json();
                expect(body.task).toBe('');
                expect(body.page).toBe('inference');
                status.target = getMockedStreamDeckTarget({ ...body });
                return HttpResponse.json(status);
            })
        );
        render(
            <StreamDeckControls
                sessionId='one'
                task=''
                page='inference'
                inference={getMockedModelControlConfig()}
                canConnect
            />
        );
        const connect = await screen.findByRole('button', { name: 'Connect Stream Deck' });
        await waitFor(() => expect(connect).toBeEnabled());
        await userEvent.setup().click(connect);
        expect(await screen.findByRole('button', { name: 'Disconnect Stream Deck' })).toBeVisible();
    });

    it('delivers each dictated draft once without overwriting subsequent manual edits on polls', async () => {
        const inference = getMockedModelControlConfig();
        status.target = getMockedStreamDeckTarget({
            page: 'inference',
            task: 'Spoken task',
            dictation: getMockedStreamDeckTranscript(),
        });
        let polls = 0;
        const updates: string[] = [];
        server.use(
            http.get('/api/streamdeck', () => {
                polls += 1;
                return HttpResponse.json(status);
            }),
            http.put('/api/streamdeck/sessions/{session_id}', async ({ request }) => {
                const body = await request.json();
                updates.push(body.task);
                status.target = getMockedStreamDeckTarget({ ...status.target, task: body.task });
                return HttpResponse.json(status);
            })
        );
        const Workspace = () => {
            const [task, setTask] = useState('Old task');
            return (
                <>
                    <TextArea label='Task draft' value={task} onChange={setTask} />
                    <StreamDeckControls
                        sessionId='one'
                        task={task}
                        page='inference'
                        inference={inference}
                        onTaskChange={setTask}
                        canConnect
                    />
                </>
            );
        };
        render(<Workspace />);
        const task = await screen.findByRole('textbox', { name: 'Task draft' });
        await waitFor(() => expect(task).toHaveValue('Spoken task'));
        expect(updates).toEqual([]);
        const user = userEvent.setup();
        await user.clear(task);
        await user.type(task, 'Manual edit');
        await waitFor(() => expect(updates).toEqual(['Manual edit']));
        await waitFor(() => expect(polls).toBeGreaterThan(1), { timeout: 2000 });
        expect(task).toHaveValue('Manual edit');
    });

    it('reports microphone activity and synchronizes changed execution settings', async () => {
        status.target = getMockedStreamDeckTarget({ page: 'inference' });
        status.dictation_status = 'listening';
        const flags: boolean[] = [];
        let hz = 0;
        server.use(
            http.put('/api/streamdeck/sessions/{session_id}', async ({ request }) => {
                const body = await request.json();
                hz = body.inference?.hz ?? 0;
                status.target = getMockedStreamDeckTarget({ ...body });
                return HttpResponse.json(status);
            })
        );
        render(
            <StreamDeckControls
                sessionId='one'
                task='Old task'
                page='inference'
                inference={getMockedModelControlConfig({ hz: 60 })}
                onDictationBusyChange={(busy) => flags.push(busy)}
                canConnect
            />
        );
        expect(await screen.findByText(/Local microphone: listening/)).toBeVisible();
        expect(flags).toContain(true);
        await waitFor(() => expect(hz).toBe(60));
    });
});
