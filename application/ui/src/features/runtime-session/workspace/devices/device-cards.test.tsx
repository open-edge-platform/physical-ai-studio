import { rs } from '@rstest/core';
import { screen, within } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaSessionState, SchemaStatusMessage } from '../../../../api/openapi-spec';
import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { getMockedProjectCamera } from '../../../../test-utils/mocks/mock-project-camera';
import {
    getMockedFollowerRobot,
    getMockedLeaderRobot,
    getMockedUnavailableRobot,
} from '../../../../test-utils/mocks/mock-robot';
import { getMockedSessionEnvironment } from '../../../../test-utils/mocks/mock-session-environment';
import { render } from '../../../../test-utils/render';
import { Statuses } from '../../runtime-session-provider';
import { DeviceCards } from './device-cards';
import { toSessionDevices } from './session-devices';

const runtime = rs.hoisted(() => ({ state: null as unknown }));

rs.mock('../../runtime-session-provider', () => ({
    useRuntimeSession: () => ({ state: runtime.state }),
}));

const PROJECT_ID = 'p1';
const follower = getMockedFollowerRobot({ id: 'follower-1', name: 'Bench arm' });
const leader = getMockedLeaderRobot({ id: 'leader-1', name: 'Teach arm' });
const frontCamera = getMockedProjectCamera({ id: 'cam-1', name: 'Front camera', driver: 'usb_camera' });

const status = (
    component: string,
    state: SchemaStatusMessage['state'] = 'running',
    error?: string
): SchemaStatusMessage => ({ type: 'status', component, state, error });

const toStatuses = (...list: SchemaStatusMessage[]): Statuses =>
    Object.fromEntries(list.map((item) => [item.component, item]));

const withLeader = getMockedSessionEnvironment({
    robot_id: follower.id,
    leader_id: leader.id,
    cameras: [{ camera_id: 'cam-1', name: 'Front', key: 'front', width: 1280, height: 720 }],
});
const followerOnly = { ...withLeader, leader_id: null };

const setState = (overrides: Partial<SchemaSessionState> = {}) => {
    runtime.state = { session_id: 's1', environment: withLeader, ...overrides };
};

const renderCards = (statuses: Statuses, environment = withLeader) =>
    render(<DeviceCards devices={toSessionDevices(statuses, environment)} />, {
        route: `/projects/${PROJECT_ID}/datasets/d1/record`,
        path: '/projects/:project_id/datasets/:dataset_id/record',
    });

const useRobots = () =>
    server.use(
        http.get('/api/projects/{project_id}/robots', () =>
            HttpResponse.json([follower, leader, getMockedUnavailableRobot()])
        ),
        http.get('/api/projects/{project_id}/cameras', () => HttpResponse.json([frontCamera]))
    );

describe('DeviceCards', () => {
    beforeEach(() => {
        setState();
        useRobots();
    });

    it('shows the follower named after its project robot, ready', async () => {
        renderCards(toStatuses(status('environment/follower')));

        expect(await screen.findByText('Robots')).toBeInTheDocument();
        expect(screen.getByText('Bench arm')).toBeInTheDocument();
        expect(screen.getByText('SO101 Follower · Follower arm')).toBeInTheDocument();
        expect(screen.getByText('Ready')).toBeInTheDocument();
        expect(screen.queryByText('Cameras')).not.toBeInTheDocument();
    });

    it('shows the follower recording while the dataset records', async () => {
        setState({
            dataset: {
                dataset_id: 'd1',
                name: 'D',
                fps: 30,
                fps_fixed: false,
                is_recording: true,
                episodes_recorded: 0,
            },
        });

        renderCards(toStatuses(status('environment/follower')));

        expect(await screen.findByText('Recording')).toBeInTheDocument();
    });

    it('shows the leader driving when the session is teleoperated', async () => {
        setState({ control: { type: 'teleop', hz: 100 } } as Partial<SchemaSessionState>);

        renderCards(toStatuses(status('environment/follower'), status('environment/leader')));

        expect(await screen.findByText('Teach arm')).toBeInTheDocument();
        expect(screen.getByText('SO101 Leader · Leader arm')).toBeInTheDocument();
        expect(screen.getByText('Driving')).toBeInTheDocument();
    });

    it('shows the leader idle when nothing is teleoperated', async () => {
        renderCards(toStatuses(status('environment/leader')));

        expect(await screen.findByText('Teach arm')).toBeInTheDocument();
        expect(screen.getByText('Idle')).toBeInTheDocument();
    });

    it('has no leader card and no placeholder name when the environment has no leader', async () => {
        renderCards(toStatuses(status('environment/follower'), status('environment/front')), followerOnly);

        expect(await screen.findByText('Bench arm')).toBeInTheDocument();
        expect(screen.queryByText('Teach arm')).not.toBeInTheDocument();
        expect(screen.queryByText(/leader arm/i)).not.toBeInTheDocument();
        expect(screen.queryByText('Leader')).not.toBeInTheDocument();
        expect(screen.getByText('Cameras')).toBeInTheDocument();
    });

    it('shows a leader without a follower status, whatever the order of the statuses', async () => {
        renderCards(
            toStatuses(status('environment/front'), status('environment/leader'), status('environment/follower'))
        );

        expect(await screen.findByText('Bench arm')).toBeInTheDocument();
        expect(screen.getByText('Teach arm')).toBeInTheDocument();
    });

    it('shows an unavailable robot as an error', async () => {
        const unavailable = getMockedUnavailableRobot({ id: 'unavailable-1' });
        server.use(http.get('/api/projects/{project_id}/robots', () => HttpResponse.json([unavailable])));

        renderCards(toStatuses(status('environment/follower')), { ...withLeader, robot_id: unavailable.id });

        expect(await screen.findByText('Missing arm')).toBeInTheDocument();
        expect(screen.getByText('Acme Follower · Follower arm (unavailable)')).toBeInTheDocument();
        expect(screen.getByText('Error')).toBeInTheDocument();
    });

    it('shows a robot the project no longer has by its id, with its status', async () => {
        server.use(http.get('/api/projects/{project_id}/robots', () => HttpResponse.json([])));

        renderCards(toStatuses(status('environment/follower')));

        expect(await screen.findByText(withLeader.robot_id)).toBeInTheDocument();
        expect(screen.getByText('Unknown robot')).toBeInTheDocument();
        expect(screen.getByText('Ready')).toBeInTheDocument();
    });

    it('shows the failure of a robot under its subtitle', async () => {
        renderCards(toStatuses(status('environment/leader', 'failed', 'Port busy')));

        expect(await screen.findByText('Teach arm')).toBeInTheDocument();
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.getByText('Port busy')).toBeInTheDocument();
    });

    it('shows a camera streaming with its driver and fps', async () => {
        renderCards(toStatuses(status('environment/front')));

        expect(await screen.findByText('Cameras')).toBeInTheDocument();
        expect(screen.getByText('Front')).toBeInTheDocument();
        expect(screen.getByText('usb_camera · 30 fps')).toBeInTheDocument();
        expect(screen.getByText('Streaming')).toBeInTheDocument();
    });

    it('shows a camera that is not in the project as unknown', async () => {
        server.use(http.get('/api/projects/{project_id}/cameras', () => HttpResponse.json([])));

        renderCards(toStatuses(status('environment/front', 'starting')));

        expect(await screen.findByText('Unknown camera')).toBeInTheDocument();
        expect(screen.getByText('Connecting')).toBeInTheDocument();
    });

    it('shows a failed camera as an error with its message', async () => {
        renderCards(toStatuses(status('environment/front', 'failed', 'Could not read a frame')));

        expect(await screen.findByText('Error')).toBeInTheDocument();
        expect(screen.getByText('Could not read a frame').closest('details')).not.toHaveAttribute('open');
        expect(screen.getByText('Error details')).toBeInTheDocument();
    });

    it('shows a leader status without a leader id as a generic card instead of crashing', async () => {
        renderCards(
            toStatuses(status('environment/follower'), status('environment/leader', 'failed', 'Odd')),
            followerOnly
        );

        expect(await screen.findByText('Other')).toBeInTheDocument();
        expect(screen.getByText('leader')).toBeInTheDocument();
        expect(screen.getByText('Odd')).toBeInTheDocument();
    });

    it('shows an unrecognised device by its raw name and state', async () => {
        renderCards(toStatuses(status('environment/mystery', 'failed', 'Nope')));

        expect(await screen.findByText('mystery')).toBeInTheDocument();
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.getByText('Nope')).toBeInTheDocument();
    });

    it('does not render a section without devices, nor anything interactive', async () => {
        const { container } = renderCards(toStatuses(status('environment/front')));

        expect(await screen.findByText('Cameras')).toBeInTheDocument();
        expect(screen.queryByText('Robots')).not.toBeInTheDocument();
        expect(within(container).queryByRole('button')).not.toBeInTheDocument();
        expect(within(container).queryByRole('link')).not.toBeInTheDocument();
    });
});
