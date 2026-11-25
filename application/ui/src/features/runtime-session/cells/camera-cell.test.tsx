import { screen } from '@testing-library/react';

import { render } from '../../../test-utils/render';
import { CameraCell } from './camera-cell';

let statuses: Record<string, { type: 'status'; component: string; state: string; error?: string | null }> = {};

rs.mock('../runtime-session-provider', () => ({
    useRuntimeSession: () => ({
        state: { environment: { cameras: [{ camera_id: 'c1', key: 'gripper', width: 640, height: 480 }] } },
        statuses,
        frame: () => ({ subscribe: () => () => undefined }),
    }),
}));

describe('CameraCell', () => {
    it('warns that the image is frozen while its camera reconnects', async () => {
        statuses = {
            'environment/gripper': {
                type: 'status',
                component: 'environment/gripper',
                state: 'starting',
                error: 'no frames for 2.0s; reconnecting',
            },
        };
        render(<CameraCell camera_id='c1' />);

        expect(await screen.findByRole('alert')).toHaveTextContent(/frozen.*reconnecting/i);
        // The last frame stays visible under the warning.
        expect(screen.getByLabelText('Camera: gripper')).toBeInTheDocument();
    });

    it('shows no warning while the camera runs', async () => {
        statuses = {
            'environment/gripper': { type: 'status', component: 'environment/gripper', state: 'running' },
        };
        render(<CameraCell camera_id='c1' />);

        expect(await screen.findByLabelText('Camera: gripper')).toBeInTheDocument();
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('shows no warning while the camera first starts', async () => {
        statuses = {
            'environment/gripper': { type: 'status', component: 'environment/gripper', state: 'starting' },
        };
        render(<CameraCell camera_id='c1' />);

        expect(await screen.findByLabelText('Camera: gripper')).toBeInTheDocument();
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
});
