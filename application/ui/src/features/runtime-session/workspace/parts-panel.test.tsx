import { rs } from '@rstest/core';
import { fireEvent, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedProjectCamera } from '../../../test-utils/mocks/mock-project-camera';
import { getMockedFollowerRobot, getMockedLeaderRobot } from '../../../test-utils/mocks/mock-robot';
import { getMockedSessionEnvironment } from '../../../test-utils/mocks/mock-session-environment';
import { render } from '../../../test-utils/render';
import { PartsPanel } from './parts-panel';

const follower = getMockedFollowerRobot({ id: 'follower-1', name: 'Bench arm' });
const leader = getMockedLeaderRobot({ id: 'leader-1', name: 'Teach arm' });

const environment = getMockedSessionEnvironment({
    name: 'Bench',
    robot_id: follower.id,
    leader_id: leader.id,
    cameras: [{ camera_id: 'camera-1', name: 'Front', key: 'front', width: 640, height: 480 }],
});

const statuses = {
    environment: { type: 'status', component: 'environment', state: 'running' },
    'environment/follower': { type: 'status', component: 'environment/follower', state: 'running' },
    'environment/leader': { type: 'status', component: 'environment/leader', state: 'failed', error: 'Unplugged' },
    'environment/front': { type: 'status', component: 'environment/front', state: 'running' },
};

const failures = rs.hoisted(() => ({ dataset: null as string | null }));

rs.mock('../runtime-session-provider', () => ({
    failureOf: (_statuses: unknown, part: string) => (part === 'dataset' ? failures.dataset : null),
    useRuntimeSession: () => ({ state: { environment }, statuses }),
}));
rs.mock('./part-controls', () => ({
    usePartControls: () => ({ pendingPart: null, act: () => undefined }),
}));

describe('PartsPanel', () => {
    afterEach(() => {
        failures.dataset = null;
    });

    it('groups the devices under the environment, and expands to show which failed and why', async () => {
        server.use(
            http.get('/api/projects/{project_id}/robots', () => HttpResponse.json([follower, leader])),
            http.get('/api/projects/{project_id}/cameras', () =>
                HttpResponse.json([getMockedProjectCamera({ id: 'camera-1', driver: 'usb_camera' })])
            )
        );

        render(<PartsPanel />, {
            route: '/projects/p1/datasets/d1/record',
            path: '/projects/:project_id/datasets/:dataset_id/record',
        });

        expect(await screen.findByText(/1 device failed/)).toBeInTheDocument();
        expect(screen.queryByText('Bench arm')).not.toBeInTheDocument();
        expect(screen.queryByText('Unplugged')).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Show devices' }));

        expect(await screen.findByText('Bench arm')).toBeInTheDocument();
        expect(screen.getByText('Ready')).toBeInTheDocument();
        expect(screen.getByText('Teach arm')).toBeInTheDocument();
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.getByText('Unplugged')).toBeInTheDocument();
        expect(screen.getByText('Front')).toBeInTheDocument();
        expect(screen.getByText('usb_camera · 30 fps')).toBeInTheDocument();
        expect(screen.getByText('Streaming')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Hide devices' })).toHaveAttribute('aria-expanded', 'true');
    });

    it('keeps a long part error inside a details element, closed until opened', async () => {
        const message = 'Could not open the dataset: '.concat('x'.repeat(300));
        failures.dataset = message;
        server.use(http.get('/api/projects/{project_id}/robots', () => HttpResponse.json([follower, leader])));

        render(<PartsPanel />, {
            route: '/projects/p1/datasets/d1/record',
            path: '/projects/:project_id/datasets/:dataset_id/record',
        });

        const summary = await screen.findByText('Error details');
        const details = summary.closest('details');
        expect(details).not.toHaveAttribute('open');
        expect(details).toHaveTextContent(message);

        fireEvent.click(summary);

        expect(details).toHaveAttribute('open');
    });
});
