import { toast } from '@geti-ui/ui';
import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaEpisode } from '../../../../api/openapi-spec';
import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { getMockedEpisodeDetail } from '../../../../test-utils/mocks/mock-episode';
import { render } from '../../../../test-utils/render';
import { Player, PlayState } from '../use-player';
import { TrimTimeline } from './trim-timeline';

const DATASET_ID = 'dataset-1';
const EPISODE_PATH = '/api/dataset/{dataset_id}/episodes/{episode_index}';
const FPS = 1;
const LENGTH = 10;

// Joint value per frame. The value changes at frames 2 and 5 only (first and last moving frame).
const MOVING_ACTIONS = [[0], [0], [10], [10], [10], [20], [20], [20], [20], [20]];
const IDLE_ACTIONS = Array.from({ length: LENGTH }, () => [0]);

// Suggested range for MOVING_ACTIONS: start = 2 - 1 = 1, end = 5 + 1 = 6.
const SUGGESTED_RANGE = { start: 1, end: 6 };
const SUGGESTED_LABEL = '1s – 6s';

const TOOLTIP_DELAY_MS = 1500;

const getMockedPlayer = (): Player => ({
    timeRef: { current: 0 },
    duration: LENGTH / FPS,
    state: PlayState.Paused,
    isPlaying: false,
    isPaused: true,
    isSeeking: false,
    play: rs.fn(),
    pause: rs.fn(),
    rewind: rs.fn(),
    seek: rs.fn(),
});

const renderTimeline = (episodeOverrides: Partial<SchemaEpisode> = {}) => {
    const episode = getMockedEpisodeDetail({
        length: LENGTH,
        fps: FPS,
        actions: MOVING_ACTIONS,
        ...episodeOverrides,
    });
    const player = getMockedPlayer();
    const user = userEvent.setup();

    render(<TrimTimeline datasetId={DATASET_ID} episode={episode} player={player} />);

    return { episode, player, user };
};

describe('TrimTimeline', () => {
    let toastInfo: ReturnType<typeof rs.spyOn>;
    let toastPositive: ReturnType<typeof rs.spyOn>;
    let toastNegative: ReturnType<typeof rs.spyOn>;

    beforeEach(() => {
        toastInfo = rs.spyOn(toast, 'info').mockImplementation(() => () => {});
        toastPositive = rs.spyOn(toast, 'positive').mockImplementation(() => () => {});
        toastNegative = rs.spyOn(toast, 'negative').mockImplementation(() => () => {});
    });

    afterEach(() => {
        toastInfo.mockRestore();
        toastPositive.mockRestore();
        toastNegative.mockRestore();
    });

    it('disables Trim for the full range', async () => {
        renderTimeline();

        expect(await screen.findByRole('button', { name: 'Trim' })).toBeDisabled();
        expect(screen.getByText(/keep 0–9 · 10\/10 frames/)).toBeInTheDocument();
    });

    it('applies the suggested range and enables Trim', async () => {
        const { user } = renderTimeline();

        await user.click(await screen.findByRole('button', { name: 'Suggest range' }));

        expect(screen.getByText(/keep 1–5 · 5\/10 frames/)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Trim' })).toBeEnabled();
    });

    it('renders a labelled track for every gripper of a bimanual robot', async () => {
        renderTimeline({
            action_keys: ['left_shoulder_pan.pos', 'left_gripper.pos', 'right_shoulder_pan.pos', 'right_gripper.pos'],
            actions: MOVING_ACTIONS.map(([value]) => [value, value, value, value]),
        });

        await screen.findByRole('button', { name: 'Trim' });

        // Each gripper name appears twice: once in the legend and once as the label of its own track.
        expect(screen.getAllByText('left gripper')).toHaveLength(2);
        expect(screen.getAllByText('right gripper')).toHaveLength(2);
        // Arm joints are only listed in the legend; they share the joint positions track.
        expect(screen.getAllByText('left shoulder pan')).toHaveLength(1);
    });

    describe('tooltip', () => {
        beforeEach(() => {
            rs.useFakeTimers({ shouldAdvanceTime: true });
        });

        afterEach(() => {
            rs.useRealTimers();
        });

        it('explains how the range is suggested', async () => {
            renderTimeline();
            const user = userEvent.setup({ advanceTimers: (ms) => rs.advanceTimersByTime(ms) });
            const button = screen.getByRole('button', { name: 'Suggest range' });

            await user.pointer({ target: document.body });
            await user.hover(button);
            expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();

            await act(() => rs.advanceTimersByTime(TOOLTIP_DELAY_MS));

            expect(screen.getByRole('tooltip')).toHaveTextContent(
                'Select frames where the robot is moving, skipping idle frames at the start and end.'
            );
        });
    });

    it('shows an info toast when no idle frames are detected', async () => {
        const { user } = renderTimeline({ actions: IDLE_ACTIONS });

        await user.click(await screen.findByRole('button', { name: 'Suggest range' }));

        expect(toastInfo).toHaveBeenCalledWith('No idle frames detected.');
        expect(screen.getByRole('button', { name: 'Trim' })).toBeDisabled();
    });

    it('sets the in and out points at the playhead and restores the full range', async () => {
        const { player, user } = renderTimeline();
        player.timeRef.current = 3;

        await user.click(await screen.findByRole('button', { name: '[ In' }));
        expect(screen.getByText(/keep 3–9/)).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Out ]' }));
        expect(screen.getByText(/keep 3–4/)).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Full' }));
        expect(screen.getByText(/keep 0–9/)).toBeInTheDocument();
    });

    it('asks for confirmation with the range in seconds and does not trim when cancelled', async () => {
        let requested = false;
        server.use(
            http.put(EPISODE_PATH, () => {
                requested = true;
                return HttpResponse.json(getMockedEpisodeDetail());
            })
        );
        const { user } = renderTimeline();
        await user.click(await screen.findByRole('button', { name: 'Suggest range' }));

        await user.click(screen.getByRole('button', { name: 'Trim' }));

        const dialog = await screen.findByRole('alertdialog');
        expect(within(dialog).getByText(new RegExp(`trim this episode to ${SUGGESTED_LABEL}`))).toBeInTheDocument();

        await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));

        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
        expect(requested).toBe(false);
        expect(toastPositive).not.toHaveBeenCalled();
    });

    it('sends the current range to the backend and shows a success toast', async () => {
        let body: unknown;
        let pathParams: unknown;
        server.use(
            http.put(EPISODE_PATH, async ({ request, params }) => {
                body = await request.json();
                pathParams = params;
                return HttpResponse.json(getMockedEpisodeDetail({ length: 5 }));
            })
        );
        const { user } = renderTimeline();
        await user.click(await screen.findByRole('button', { name: 'Suggest range' }));

        await user.click(screen.getByRole('button', { name: 'Trim' }));
        const dialog = await screen.findByRole('alertdialog');
        await user.click(within(dialog).getByRole('button', { name: 'Trim' }));

        await waitFor(() => expect(toastPositive).toHaveBeenCalledWith('Episode trimmed.'));
        expect(body).toEqual({ start_frame: SUGGESTED_RANGE.start, end_frame: SUGGESTED_RANGE.end });
        expect(pathParams).toEqual({ dataset_id: DATASET_ID, episode_index: '0' });
        expect(toastNegative).not.toHaveBeenCalled();
    });

    it('shows an error toast when the backend rejects the trim', async () => {
        server.use(
            http.put(EPISODE_PATH, () =>
                HttpResponse.json(
                    { detail: [{ loc: ['body', 'end_frame'], msg: 'end_frame exceeds length', type: 'value_error' }] },
                    { status: 422 }
                )
            )
        );
        const { user } = renderTimeline();
        await user.click(await screen.findByRole('button', { name: 'Suggest range' }));

        await user.click(screen.getByRole('button', { name: 'Trim' }));
        await user.click(within(await screen.findByRole('alertdialog')).getByRole('button', { name: 'Trim' }));

        await waitFor(() => expect(toastNegative).toHaveBeenCalledWith('end_frame: end_frame exceeds length'));
        expect(toastPositive).not.toHaveBeenCalled();
    });
});
