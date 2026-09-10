import { ToastQueue } from '@geti-ui/ui';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaRuntimeSessionInfo } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { RuntimeSessionsDialog, RuntimeSessionStatus } from './runtime-sessions';
import { sessionStatusVariant } from './use-runtime-sessions';

const SESSIONS_PATH = '/api/runtime/sessions';
const COUNT_PATH = '/api/runtime/sessions/count';
const STOP_PATH = '/api/runtime/sessions/{session_name}/stop';

const ROBOT_ID = '9c1f5b8e-2f4a-4c6e-8a11-7d2e3f4a5b60';

const baseState: NonNullable<SchemaRuntimeSessionInfo['activity']> = {
    connected: true,
    follower_source: 'teleop',
    model_loaded: false,
    task: 'pick up the cube',
    dataset_loaded: true,
    is_recording: false,
    episodes_recorded: 3,
};

const session = (overrides: Partial<SchemaRuntimeSessionInfo> = {}): SchemaRuntimeSessionInfo => ({
    session_name: `rt-${ROBOT_ID}`,
    follower_id: ROBOT_ID,
    status: 'running',
    pid: 41273,
    follower_name: 'left arm',
    leader_name: 'left leader',
    started_at: new Date(Date.now() - 90_000).toISOString(),
    idle_timeout_s: 45,
    attached: true,
    idle_deadline: null,
    camera_keys: ['wrist'],
    activity: baseState,
    error: null,
    ...overrides,
});

describe('RuntimeSessionStatus', () => {
    afterEach(() => {
        Object.defineProperty(window.screen, 'width', { configurable: true, value: 0 });
    });

    it('renders nothing when no sessions are running', async () => {
        // The footer is mounted on every page, so the resting state is this one.
        let answered = 0;
        server.use(
            http.get(COUNT_PATH, () => {
                answered += 1;
                return HttpResponse.json({ count: 0 });
            })
        );

        const { container } = render(<RuntimeSessionStatus />);

        // Wait for the answer first, or this would pass before the query resolves.
        await waitFor(() => expect(answered).toBeGreaterThan(0));
        expect(container.querySelector('#theme-provider')).toBeEmptyDOMElement();
    });

    it('names one session in the singular', async () => {
        server.use(http.get(COUNT_PATH, () => HttpResponse.json({ count: 1 })));

        render(<RuntimeSessionStatus />);

        expect(await screen.findByText('1 session')).toBeInTheDocument();
    });

    it('counts concurrent sessions', async () => {
        server.use(http.get(COUNT_PATH, () => HttpResponse.json({ count: 3 })));

        render(<RuntimeSessionStatus />);

        expect(await screen.findByText('3 sessions')).toBeInTheDocument();
    });

    it('opens the session list from the footer chip', async () => {
        const user = userEvent.setup();
        server.use(
            http.get(COUNT_PATH, () => HttpResponse.json({ count: 1 })),
            http.get(SESSIONS_PATH, () => HttpResponse.json([session()]))
        );

        render(<RuntimeSessionStatus />);

        await user.click(await screen.findByRole('button', { name: 'Runtime sessions' }));

        expect(await screen.findByRole('heading', { name: 'Runtime sessions' })).toBeInTheDocument();
        expect(await screen.findByText('left arm')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Close' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Pin session list' })).toBeInTheDocument();
    });

    it('stays open when pinned if the user clicks away or presses Escape', async () => {
        const user = userEvent.setup();
        const outside = vi.fn();
        // Spectrum swaps a popover for a modal at <= 700px. jsdom's screen is
        // 0, so without this the pin path never hits the non-modal popover.
        Object.defineProperty(window.screen, 'width', { configurable: true, value: 1280 });
        server.use(
            http.get(COUNT_PATH, () => HttpResponse.json({ count: 1 })),
            http.get(SESSIONS_PATH, () => HttpResponse.json([session()]))
        );

        render(
            <>
                <button type='button' onClick={outside}>
                    Switch fixture
                </button>
                <RuntimeSessionStatus />
            </>
        );

        await user.click(await screen.findByRole('button', { name: 'Runtime sessions' }));

        await user.click(await screen.findByRole('button', { name: 'Pin session list' }));

        expect(screen.getByRole('button', { name: 'Unpin session list' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.queryByTestId('underlay')).not.toBeInTheDocument();

        await user.keyboard('{Escape}');
        await user.click(screen.getByRole('button', { name: 'Switch fixture' }));

        expect(outside).toHaveBeenCalled();
        expect(screen.getByRole('heading', { name: 'Runtime sessions' })).toBeInTheDocument();
    });

    it('keeps an expanded session visible after pinning', async () => {
        const user = userEvent.setup();
        Object.defineProperty(window.screen, 'width', { configurable: true, value: 1280 });
        server.use(
            http.get(COUNT_PATH, () => HttpResponse.json({ count: 1 })),
            http.get(SESSIONS_PATH, () => HttpResponse.json([session()]))
        );

        render(<RuntimeSessionStatus />);

        await user.click(await screen.findByRole('button', { name: 'Runtime sessions' }));
        await user.click(await screen.findByRole('button', { name: /details for left arm/i }));

        const heading = await screen.findByRole('heading', { name: 'Runtime sessions' });
        const task = screen.getByText('pick up the cube');

        await user.click(screen.getByRole('button', { name: 'Pin session list' }));

        expect(screen.getByRole('button', { name: 'Unpin session list' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('heading', { name: 'Runtime sessions' })).toBe(heading);
        expect(screen.getByText('pick up the cube')).toBe(task);
        expect(screen.getByRole('button', { name: /details for left arm/i })).toHaveAttribute('aria-expanded', 'true');
    });

    it('closes when clicking away if it is not pinned', async () => {
        const user = userEvent.setup();
        Object.defineProperty(window.screen, 'width', { configurable: true, value: 1280 });
        server.use(
            http.get(COUNT_PATH, () => HttpResponse.json({ count: 1 })),
            http.get(SESSIONS_PATH, () => HttpResponse.json([session()]))
        );

        render(
            <>
                <button type='button'>Switch fixture</button>
                <RuntimeSessionStatus />
            </>
        );

        await user.click(await screen.findByRole('button', { name: 'Runtime sessions' }));
        expect(await screen.findByRole('heading', { name: 'Runtime sessions' })).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Switch fixture' }));

        expect(screen.getByRole('button', { name: 'Runtime sessions' })).toHaveAttribute('aria-expanded', 'false');
    });

    it('keeps a pinned list open when the last session stops', async () => {
        const user = userEvent.setup();
        let count = 1;
        let sessions = [session()];
        server.use(
            http.get(COUNT_PATH, () => HttpResponse.json({ count })),
            http.get(SESSIONS_PATH, () => HttpResponse.json(sessions)),
            http.post(STOP_PATH, () => {
                count = 0;
                sessions = [];
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<RuntimeSessionStatus />);

        await user.click(await screen.findByRole('button', { name: 'Runtime sessions' }));
        await user.click(await screen.findByRole('button', { name: 'Pin session list' }));
        await user.click(screen.getByRole('button', { name: /stop session for left arm/i }));
        await user.click(screen.getByRole('button', { name: 'Stop session' }));

        // The chip unmounting at zero would take the popover with it, so a pinned
        // panel would vanish on the very action the user opened it to perform.
        expect(await screen.findByText('No runtime sessions are running.')).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Runtime sessions' })).toBeInTheDocument();
    });

    it('closes a pinned list from the header close button', async () => {
        const user = userEvent.setup();
        server.use(
            http.get(COUNT_PATH, () => HttpResponse.json({ count: 1 })),
            http.get(SESSIONS_PATH, () => HttpResponse.json([session()]))
        );

        render(<RuntimeSessionStatus />);

        await user.click(await screen.findByRole('button', { name: 'Runtime sessions' }));
        await user.click(await screen.findByRole('button', { name: 'Pin session list' }));
        await user.click(screen.getByRole('button', { name: 'Close' }));

        await waitFor(() =>
            expect(screen.queryByRole('heading', { name: 'Runtime sessions' })).not.toBeInTheDocument()
        );
    });
});

describe('RuntimeSessionsDialog', () => {
    it('shows an empty state when nothing is running', async () => {
        server.use(http.get(SESSIONS_PATH, () => HttpResponse.json([])));

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        expect(await screen.findByText('No runtime sessions are running.')).toBeInTheDocument();
    });

    it('lists a running session with what it is doing', async () => {
        server.use(http.get(SESSIONS_PATH, () => HttpResponse.json([session()])));

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        expect(await screen.findByText('left arm')).toBeInTheDocument();
        expect(screen.getByText('teleop')).toBeInTheDocument();
    });

    it('says a session is recording rather than merely running', async () => {
        server.use(
            http.get(SESSIONS_PATH, () =>
                HttpResponse.json([session({ activity: { ...baseState, is_recording: true } })])
            )
        );

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        expect(await screen.findByText('recording')).toBeInTheDocument();
    });

    it('warns that nobody is watching an abandoned session', async () => {
        server.use(
            http.get(SESSIONS_PATH, () =>
                HttpResponse.json([
                    session({ attached: false, idle_deadline: new Date(Date.now() + 30_000).toISOString() }),
                ])
            )
        );
        const user = userEvent.setup();

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        await user.click(await screen.findByRole('button', { name: /details for left arm/i }));

        expect(await screen.findByText(/nobody is watching this session/i)).toBeInTheDocument();
    });

    it('labels expanded session details so the fields are readable', async () => {
        const user = userEvent.setup();
        server.use(http.get(SESSIONS_PATH, () => HttpResponse.json([session()])));

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        await user.click(await screen.findByRole('button', { name: /details for left arm/i }));

        expect(screen.getByText('Task')).toBeInTheDocument();
        expect(screen.getByText('pick up the cube')).toBeInTheDocument();
        expect(screen.getByText('Dataset')).toBeInTheDocument();
        expect(screen.getByText('Loaded')).toBeInTheDocument();
        expect(screen.getByText('Model')).toBeInTheDocument();
        expect(screen.getByText('Not loaded')).toBeInTheDocument();
        expect(screen.getByText('Leader robot')).toBeInTheDocument();
        expect(screen.getByText('left leader')).toBeInTheDocument();
        expect(screen.getByText('Cameras')).toBeInTheDocument();
        expect(screen.getByText('wrist')).toBeInTheDocument();
        expect(screen.getByText('Process ID')).toBeInTheDocument();
        expect(screen.getByText('41273')).toBeInTheDocument();
        expect(screen.getByText('Session ID')).toBeInTheDocument();
        expect(screen.getByText(`rt-${ROBOT_ID}`)).toBeInTheDocument();
    });

    it('lists a session that will not answer instead of hiding it', async () => {
        server.use(
            http.get(SESSIONS_PATH, () =>
                HttpResponse.json([
                    {
                        session_name: `rt-${ROBOT_ID}`,
                        follower_id: ROBOT_ID,
                        status: 'unreachable',
                        pid: 41402,
                        camera_keys: [],
                        activity: null,
                        error: null,
                    },
                ])
            )
        );

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        // Falls back to the raw session name: there is no label to show.
        expect(await screen.findByText(`rt-${ROBOT_ID}`)).toBeInTheDocument();
        expect(screen.getByText('unreachable')).toBeInTheDocument();
    });

    it('stops a session after confirming', async () => {
        const user = userEvent.setup();
        let sessions = [session()];
        const stopped: string[] = [];
        server.use(
            http.get(SESSIONS_PATH, () => HttpResponse.json(sessions)),
            http.post(STOP_PATH, ({ params }) => {
                stopped.push(String(params.session_name));
                sessions = [];
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        await user.click(await screen.findByRole('button', { name: /stop session for left arm/i }));
        expect(await screen.findByRole('heading', { name: /stop session for 'left arm'\?/i })).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Stop session' }));

        await waitFor(() => expect(stopped).toEqual([`rt-${ROBOT_ID}`]));
        expect(await screen.findByText('No runtime sessions are running.')).toBeInTheDocument();
    });

    it('keeps the session when the confirmation is cancelled', async () => {
        const user = userEvent.setup();
        const stopped: string[] = [];
        server.use(
            http.get(SESSIONS_PATH, () => HttpResponse.json([session()])),
            http.post(STOP_PATH, ({ params }) => {
                stopped.push(String(params.session_name));
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        await user.click(await screen.findByRole('button', { name: /stop session for left arm/i }));
        await user.click(await screen.findByRole('button', { name: 'Cancel' }));

        await waitFor(() => expect(screen.queryByRole('button', { name: 'Stop session' })).not.toBeInTheDocument());
        expect(stopped).toEqual([]);
    });

    it('returns focus to the Stop button when the confirmation is cancelled', async () => {
        const user = userEvent.setup();
        server.use(http.get(SESSIONS_PATH, () => HttpResponse.json([session()])));

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        const stop = await screen.findByRole('button', { name: /stop session for left arm/i });
        await user.click(stop);

        // Focus moves to the safe option, never the destructive one.
        expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();

        await user.click(screen.getByRole('button', { name: 'Cancel' }));

        // Back where it started. Unmounting the Stop button instead would drop
        // focus to the body and strand a keyboard user.
        await waitFor(() => expect(stop).toHaveFocus());
    });

    it('cancels the confirmation on Escape without closing the list', async () => {
        const user = userEvent.setup();
        server.use(http.get(SESSIONS_PATH, () => HttpResponse.json([session()])));

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        await user.click(await screen.findByRole('button', { name: /stop session for left arm/i }));
        await user.keyboard('{Escape}');

        await waitFor(() => expect(screen.queryByRole('button', { name: 'Stop session' })).not.toBeInTheDocument());
        expect(screen.getByRole('heading', { name: 'Runtime sessions' })).toBeInTheDocument();
    });

    it('lets a second session be stopped while the first stop is still in flight', async () => {
        const user = userEvent.setup();
        server.use(
            http.get(SESSIONS_PATH, () =>
                HttpResponse.json([
                    session(),
                    session({ session_name: 'rt-second', follower_name: 'right arm', follower_id: null }),
                ])
            ),
            // Never settles, standing in for the seconds the backend spends
            // waiting out SIGTERM before it escalates to SIGKILL.
            http.post(STOP_PATH, () => new Promise<never>(() => {}))
        );

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        await user.click(await screen.findByRole('button', { name: /stop session for left arm/i }));
        await user.click(screen.getByRole('button', { name: 'Stop session' }));

        await user.click(await screen.findByRole('button', { name: /stop session for right arm/i }));

        // A pending stop on one session must not disable the confirmation on
        // another, or the second robot cannot be stopped until the first lets go.
        expect(await screen.findByRole('button', { name: 'Stop session' })).toBeEnabled();
    });

    it('surfaces a session that would not stop', async () => {
        const user = userEvent.setup();
        // The primary action dismisses the dialog, so the failure has to land
        // somewhere that outlives it. Toasts render into a container the test
        // wrapper does not mount, so assert on the queue itself.
        const toast = vi.spyOn(ToastQueue, 'negative').mockImplementation(() => () => {});
        server.use(
            http.get(SESSIONS_PATH, () => HttpResponse.json([session()])),
            // Needs a body: the spec declares no error envelope, and a bodyless
            // response is indistinguishable from success to the generated
            // client. `{ detail: [] }` is what the other suites use.
            http.post(STOP_PATH, () => HttpResponse.json({ detail: [] }, { status: 500 }))
        );

        render(<RuntimeSessionsDialog close={vi.fn()} />);

        await user.click(await screen.findByRole('button', { name: /stop session for left arm/i }));
        await user.click(await screen.findByRole('button', { name: 'Stop session' }));

        await waitFor(() => expect(toast).toHaveBeenCalledWith(expect.stringMatching(/could not be stopped/i)));
        toast.mockRestore();
    });
});

describe('sessionStatusVariant', () => {
    it('uses yellow for hold and green for teleop', () => {
        expect(sessionStatusVariant(session({ activity: { ...baseState, follower_source: 'hold' } }))).toBe('notice');
        expect(sessionStatusVariant(session())).toBe('positive');
        expect(sessionStatusVariant(session({ activity: { ...baseState, is_recording: true } }))).toBe('positive');
    });

    it('uses red for broken sessions', () => {
        expect(sessionStatusVariant(session({ status: 'unreachable', activity: null }))).toBe('negative');
        expect(sessionStatusVariant(session({ status: 'error' }))).toBe('negative');
    });
});
