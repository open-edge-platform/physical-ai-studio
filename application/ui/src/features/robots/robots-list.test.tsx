import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { RobotsList } from './robots-list';

const PROJECT_ID = 'test-project-id';
const ROBOT_ID = 'robot-id';
const CALIBRATION_ID = 'calibration-id';
const ROBOTS_PATH = '/api/projects/{project_id}/robots';
const ONLINE_ROBOTS_PATH = '/api/projects/{project_id}/robots/online';

const so101Robot = {
    id: ROBOT_ID,
    name: 'Test SO101',
    type: 'SO101_Follower' as const,
    payload: { connection_string: '', serial_number: 'SO101-001' },
    active_calibration_id: CALIBRATION_ID,
};

const renderRobotsList = () =>
    render(<RobotsList />, {
        route: `/projects/${PROJECT_ID}/robots`,
        path: '/projects/:project_id/robots',
    });

const openRobotMenu = async (user: ReturnType<typeof userEvent.setup>) => {
    await screen.findByText(so101Robot.name);
    await user.click(screen.getByRole('button', { name: `Actions for ${so101Robot.name}` }));
};

describe('RobotsList', () => {
    afterEach(() => {
        vi.restoreAllMocks();
        vi.unstubAllGlobals();
    });

    it('shows Export calibration for SO101 robots', async () => {
        server.use(
            http.get(ROBOTS_PATH, () =>
                HttpResponse.json([
                    so101Robot,
                    {
                        id: 'widowx-id',
                        name: 'Test WidowX',
                        type: 'Trossen_WidowXAI_Follower',
                        payload: { connection_string: '', serial_number: 'widowx-001' },
                    },
                ])
            ),
            http.get(ONLINE_ROBOTS_PATH, () => HttpResponse.json([]))
        );

        const user = userEvent.setup();
        renderRobotsList();

        await openRobotMenu(user);

        expect(await screen.findByText('Export calibration', { selector: '[role]' })).toBeEnabled();
    });

    it('disables Export calibration when no calibration is active', async () => {
        server.use(
            http.get(ROBOTS_PATH, () => HttpResponse.json([{ ...so101Robot, active_calibration_id: null }])),
            http.get(ONLINE_ROBOTS_PATH, () => HttpResponse.json([]))
        );

        const user = userEvent.setup();
        renderRobotsList();

        await openRobotMenu(user);

        const exportCalibration = await screen.findByText('Export calibration', { selector: '[role]' });
        expect(exportCalibration.closest('[aria-disabled]')).toHaveAttribute('aria-disabled', 'true');
    });

    it('downloads the active calibration in import-compatible format', async () => {
        const NativeURL = URL;
        const createObjectUrl = vi.fn<(blob: Blob) => string>(() => 'blob:calibration');
        const revokeObjectUrl = vi.fn();
        const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
        vi.stubGlobal(
            'URL',
            class extends NativeURL {
                static createObjectURL = createObjectUrl;
                static revokeObjectURL = revokeObjectUrl;
            }
        );
        server.use(
            http.get(ROBOTS_PATH, () => HttpResponse.json([so101Robot])),
            http.get(ONLINE_ROBOTS_PATH, () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/robots/{robot_id}/calibrations/{calibration_id}', () =>
                HttpResponse.json({
                    id: CALIBRATION_ID,
                    robot_id: ROBOT_ID,
                    file_path: '/calibration.json',
                    values: {
                        shoulder_pan: {
                            id: 1,
                            joint_name: 'shoulder_pan',
                            drive_mode: 0,
                            homing_offset: 10,
                            range_min: -100,
                            range_max: 100,
                        },
                    },
                })
            )
        );

        const user = userEvent.setup();
        renderRobotsList();

        await openRobotMenu(user);
        await user.click(await screen.findByText('Export calibration', { selector: '[role]' }));

        await waitFor(() => expect(createObjectUrl).toHaveBeenCalledOnce());

        const blob = createObjectUrl.mock.calls[0]?.[0];
        expect(blob).toBeDefined();
        expect(JSON.parse(await blob.text())).toEqual({
            shoulder_pan: {
                id: 1,
                drive_mode: 0,
                homing_offset: 10,
                range_min: -100,
                range_max: 100,
            },
        });
        expect(click).toHaveBeenCalledOnce();
        expect(revokeObjectUrl).toHaveBeenCalledWith('blob:calibration');
    });
});
