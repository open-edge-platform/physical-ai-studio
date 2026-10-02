import { createRef } from 'react';

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SchemaModel } from '../../../api/openapi-spec';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { render } from '../../../test-utils/render';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { InferenceViewer } from './inference-viewer';

vi.mock('../../robots/runtime-session-provider', () => ({
    useRuntimeSession: vi.fn(),
}));

vi.mock('../../robots/robot-control/robot-control-view', () => ({
    RobotControlView: () => <div>robot-control</div>,
}));

const PROJECT_ID = 'project-1';

const model = {
    id: 'model-1',
    name: 'ACT',
    path: '/models/act',
    policy: 'act',
    properties: {},
    project_id: PROJECT_ID,
    dataset_id: 'dataset-1',
    snapshot_id: null,
    train_job_id: 'job-1',
    parent_model_id: null,
    version: 1,
    created_at: '2026-07-14T12:00:00Z',
    available_backends: ['pytorch'],
    lora_enabled: false,
    lora_use_dora: false,
    snapflow_enabled: false,
} as SchemaModel;

const followerRobot = {
    id: 'follower-1',
    name: 'Follower',
    type: 'SO101_Follower' as const,
    payload: { connection_string: '', serial_number: 'SO101-001', calibration: {} },
};

const environmentWithLeader = getMockedEnvironment({
    robots: [
        {
            robot: followerRobot,
            tele_operator: { type: 'robot', robot_id: 'leader-1' },
        },
    ],
});

const mutation = () => ({
    mutate: vi.fn(),
    isPending: false,
});

const mockSession = ({
    followerSource = 'hold',
    hasLeader = true,
    readyForInference = true,
}: {
    followerSource?: 'hold' | 'teleop' | 'policy';
    hasLeader?: boolean;
    readyForInference?: boolean;
} = {}) => {
    const startTask = mutation();
    const stopTask = mutation();
    const setFollowerSource = mutation();

    vi.mocked(useRuntimeSession).mockReturnValue({
        model,
        readyForInference,
        state: {
            connected: true,
            follower_source: followerSource,
            has_leader: hasLeader,
            model_loaded: true,
            task: null,
            dataset_loaded: false,
            is_recording: false,
            episodes_recorded: 0,
        },
        startTask,
        stopTask,
        setFollowerSource,
        dataset: undefined,
        loadDataset: mutation(),
        startEpisode: mutation(),
        saveEpisode: mutation(),
        discardEpisode: mutation(),
        environment: environmentWithLeader,
        observation: createRef<Record<string, number> | undefined>(),
        inferenceDevice: { backend: 'pytorch', device: 'cpu' },
    } as unknown as ReturnType<typeof useRuntimeSession>);

    return { startTask, stopTask, setFollowerSource };
};

const renderViewer = (tasks: string[] = ['pick the cube']) =>
    render(<InferenceViewer tasks={tasks} />, {
        route: `/projects/${PROJECT_ID}/models/model-1/inference/pytorch`,
        path: '/projects/:project_id/models/:model_id/inference/:backend',
    });

describe('InferenceViewer', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('enables the teleoperate toggle when the session has a leader robot', async () => {
        mockSession();

        renderViewer();

        expect(await screen.findByRole('switch', { name: 'Teleoperate' })).not.toBeChecked();
        expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('disables the teleoperate toggle when the session has no leader robot', async () => {
        mockSession({ hasLeader: false });

        renderViewer();

        expect(await screen.findByRole('button', { name: /play/i })).toBeInTheDocument();
        expect(screen.getByRole('switch', { name: 'Teleoperate' })).toBeDisabled();
    });

    it('selects the toggle while teleoperating', async () => {
        mockSession({ followerSource: 'teleop' });

        renderViewer();

        expect(await screen.findByRole('switch', { name: 'Teleoperate' })).toBeChecked();
        expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('leaves the toggle off during policy inference', async () => {
        mockSession({ followerSource: 'policy' });

        renderViewer();

        expect(await screen.findByRole('switch', { name: 'Teleoperate' })).not.toBeChecked();
        expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
    });

    it('switches the follower source to teleop and hold', async () => {
        const user = userEvent.setup();
        const { setFollowerSource } = mockSession();

        renderViewer();

        await user.click(await screen.findByRole('switch', { name: 'Teleoperate' }));

        expect(setFollowerSource.mutate).toHaveBeenCalledWith('teleop');
    });

    it('switches back to hold when teleoperation is turned off', async () => {
        const user = userEvent.setup();
        const { setFollowerSource } = mockSession({ followerSource: 'teleop' });

        renderViewer();

        await user.click(await screen.findByRole('switch', { name: 'Teleoperate' }));

        expect(setFollowerSource.mutate).toHaveBeenCalledWith('hold');
    });

    it('takes over from policy inference when teleoperate is enabled', async () => {
        const user = userEvent.setup();
        const { setFollowerSource, stopTask } = mockSession({ followerSource: 'policy' });

        renderViewer();

        await user.click(await screen.findByRole('switch', { name: 'Teleoperate' }));

        expect(setFollowerSource.mutate).toHaveBeenCalledWith('teleop');
        expect(stopTask.mutate).not.toHaveBeenCalled();
    });
});

describe('InferenceViewer task prompt', () => {
    const originalScreenWidth = window.screen.width;

    beforeEach(() => {
        vi.clearAllMocks();
        // React Spectrum renders the mobile combo box on narrow screens; jsdom reports width 0.
        Object.defineProperty(window.screen, 'width', { value: 1920, configurable: true });
    });

    afterEach(() => {
        Object.defineProperty(window.screen, 'width', { value: originalScreenWidth, configurable: true });
    });

    it('keeps the task after the prompt field loses focus', async () => {
        const { startTask } = mockSession();
        renderViewer();

        const field = await screen.findByRole('combobox', { name: 'Task prompt' });
        expect(field).toHaveValue('pick the cube');

        await userEvent.click(field);
        await userEvent.click(screen.getByRole('button', { name: /play/i }));

        expect(field).toHaveValue('pick the cube');
        expect(startTask.mutate).toHaveBeenCalledWith('pick the cube');
    });

    it('sends the task picked from the list and keeps showing it', async () => {
        const { startTask } = mockSession();
        renderViewer(['pick the cube', 'stack the blocks']);

        const field = await screen.findByRole('combobox', { name: 'Task prompt' });
        await userEvent.click(screen.getByRole('button', { name: /show suggestions/i }));
        await userEvent.click(await screen.findByRole('option', { name: 'stack the blocks' }));
        await userEvent.click(screen.getByRole('button', { name: /play/i }));

        expect(field).toHaveValue('stack the blocks');
        expect(startTask.mutate).toHaveBeenCalledWith('stack the blocks');
    });

    it('sends a custom prompt typed into the field', async () => {
        const { startTask } = mockSession();
        renderViewer();

        const field = await screen.findByRole('combobox', { name: 'Task prompt' });
        await userEvent.clear(field);
        await userEvent.type(field, 'put the red cube in the box');
        await userEvent.click(screen.getByRole('button', { name: /play/i }));

        expect(field).toHaveValue('put the red cube in the box');
        expect(startTask.mutate).toHaveBeenCalledWith('put the red cube in the box');
    });

    it('asks for confirmation when the prompt is cleared', async () => {
        const { startTask } = mockSession();
        renderViewer();

        const field = await screen.findByRole('combobox', { name: 'Task prompt' });
        await userEvent.clear(field);
        await userEvent.keyboard('{Escape}');
        await userEvent.tab();
        await userEvent.click(screen.getByRole('button', { name: /play/i }));

        expect(await screen.findByRole('alertdialog')).toHaveTextContent('Start without a task prompt?');
        expect(startTask.mutate).not.toHaveBeenCalled();
    });

    it('asks for confirmation before starting with an empty prompt', async () => {
        const { startTask } = mockSession();
        renderViewer([]);

        await userEvent.click(await screen.findByRole('button', { name: /play/i }));

        expect(await screen.findByRole('alertdialog')).toHaveTextContent('Start without a task prompt?');
        expect(startTask.mutate).not.toHaveBeenCalled();

        await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
        expect(startTask.mutate).not.toHaveBeenCalled();

        await userEvent.click(screen.getByRole('button', { name: /play/i }));
        await userEvent.click(await screen.findByRole('button', { name: 'Start anyway' }));
        expect(startTask.mutate).toHaveBeenCalledWith('');
    });
});
