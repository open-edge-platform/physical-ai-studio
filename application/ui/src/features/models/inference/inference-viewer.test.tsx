import { createRef } from 'react';

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { SchemaEnvironmentWithRelations, SchemaModel } from '../../../api/openapi-spec';
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

const environmentWithoutLeader = getMockedEnvironment({
    robots: [
        {
            robot: followerRobot,
            tele_operator: { type: 'none' },
        },
    ],
});

const mutation = () => ({
    mutate: vi.fn(),
    isPending: false,
});

const mockSession = ({
    followerSource = 'hold',
    environment = environmentWithLeader,
    readyForInference = true,
}: {
    followerSource?: 'hold' | 'teleop' | 'policy';
    environment?: SchemaEnvironmentWithRelations;
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
            model_loaded: true,
            task: null,
            dataset_loaded: false,
            is_recording: false,
            episodes_recorded: 0,
        },
        startTask,
        stopTask,
        setFollowerSource,
        environment,
        observation: createRef<Record<string, number> | undefined>(),
        inferenceDevice: { backend: 'pytorch', device: 'cpu' },
    } as unknown as ReturnType<typeof useRuntimeSession>);

    return { startTask, stopTask, setFollowerSource };
};

const renderViewer = () =>
    render(<InferenceViewer tasks={['pick the cube']} />, {
        route: `/projects/${PROJECT_ID}/models/model-1/inference/pytorch`,
        path: '/projects/:project_id/models/:model_id/inference/:backend',
    });

describe('InferenceViewer', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('shows a teleoperate toggle when the environment has a leader robot', async () => {
        mockSession();

        renderViewer();

        expect(await screen.findByRole('switch', { name: 'Teleoperate' })).not.toBeChecked();
        expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('hides the teleoperate toggle when the environment has no leader robot', async () => {
        mockSession({ environment: environmentWithoutLeader });

        renderViewer();

        expect(await screen.findByRole('button', { name: /play/i })).toBeInTheDocument();
        expect(screen.queryByRole('switch', { name: 'Teleoperate' })).not.toBeInTheDocument();
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
