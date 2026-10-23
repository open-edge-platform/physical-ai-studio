import { createRef } from 'react';

import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { useRuntimeSession } from '../../../features/robots/runtime-session-provider';
import { FollowerSource } from '../../../features/robots/use-joint-state';
import { server } from '../../../msw-node-setup';
import { getMockedDataset } from '../../../test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { render } from '../../../test-utils/render';
import { RecordingViewer } from './recording-viewer';

rs.mock('../../../features/robots/runtime-session-provider', () => ({
    useRuntimeSession: rs.fn(),
}));

rs.mock('../../../features/robots/robot-control/robot-control-view', () => ({
    RobotControlView: () => <div>robot-control</div>,
}));

const mutation = () => ({
    mutate: rs.fn(),
    isPending: false,
});

const mockSession = ({
    followerSource = 'teleop',
    hasLeader = true,
    isRecording = false,
}: {
    followerSource?: FollowerSource;
    hasLeader?: boolean;
    isRecording?: boolean;
} = {}) => {
    const setFollowerSource = mutation();

    rs.mocked(useRuntimeSession).mockReturnValue({
        dataset: getMockedDataset(),
        readyForRecording: true,
        state: {
            connected: true,
            follower_source: followerSource,
            has_leader: hasLeader,
            model_loaded: false,
            task: null,
            dataset_loaded: true,
            is_recording: isRecording,
            episodes_recorded: 0,
        },
        startEpisode: mutation(),
        saveEpisode: mutation(),
        discardEpisode: mutation(),
        setFollowerSource,
        environment: getMockedEnvironment(),
        observation: createRef<Record<string, number> | undefined>(),
        actions: createRef<Record<string, number> | undefined>(),
    } as unknown as ReturnType<typeof useRuntimeSession>);

    return { setFollowerSource };
};

const renderViewer = () => {
    server.use(http.get('/api/settings', () => HttpResponse.json({ hotkeys: { bindings: {} } } as never)));
    return render(<RecordingViewer />);
};

describe('RecordingViewer', () => {
    beforeEach(() => {
        rs.clearAllMocks();
    });

    it('switches to external control with a leader', async () => {
        const { setFollowerSource } = mockSession();

        renderViewer();

        const toggle = await screen.findByRole('switch', { name: 'External control' });
        expect(toggle).not.toBeChecked();
        await userEvent.setup().click(toggle);
        expect(setFollowerSource.mutate).toHaveBeenCalledWith('observe');
    });

    it('offers external control without a leader', async () => {
        const { setFollowerSource } = mockSession({ followerSource: 'hold', hasLeader: false });

        renderViewer();

        const toggle = await screen.findByRole('switch', { name: 'External control' });
        expect(toggle).toBeEnabled();
        await userEvent.setup().click(toggle);
        expect(setFollowerSource.mutate).toHaveBeenCalledWith('observe');
    });

    it('returns to the leader when external control is turned off', async () => {
        const { setFollowerSource } = mockSession({ followerSource: 'observe' });

        renderViewer();

        const toggle = await screen.findByRole('switch', { name: 'External control' });
        expect(toggle).toBeChecked();
        await userEvent.setup().click(toggle);
        expect(setFollowerSource.mutate).toHaveBeenCalledWith('teleop');
    });

    it('holds when external control is turned off without a leader', async () => {
        const { setFollowerSource } = mockSession({ followerSource: 'observe', hasLeader: false });

        renderViewer();

        await userEvent.setup().click(await screen.findByRole('switch', { name: 'External control' }));
        expect(setFollowerSource.mutate).toHaveBeenCalledWith('hold');
    });

    it('locks the control choice while an episode is recording', async () => {
        mockSession({ followerSource: 'observe', isRecording: true });

        renderViewer();

        expect(await screen.findByRole('switch', { name: 'External control' })).toBeDisabled();
    });
});
