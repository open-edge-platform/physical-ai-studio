import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { render } from '../../../../test-utils/render';
import { RobotFormProvider, useRobotForm } from '../provider';
import { BimanualSO101FormFields } from './bimanual-so101';

const PROJECT_ID = 'test-project-id';
const ROBOTS_PATH = '/api/projects/{project_id}/robots';

const calibration = {
    shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 0, range_min: 0, range_max: 4095 },
};

const so101Robot = <T extends 'SO101_Follower' | 'SO101_Leader'>(
    type: T,
    id: string,
    name: string,
    serialNumber: string,
    withCalibration: boolean = true
) => ({
    id,
    name,
    type,
    payload: {
        connection_string: '',
        serial_number: serialNumber,
        calibration: withCalibration ? calibration : null,
    },
});

const Payload = () => {
    const { payload } = useRobotForm();
    return <output>{JSON.stringify(payload)}</output>;
};

const renderFields = (activeType: string, payload: Record<string, unknown> = {}) =>
    render(
        <RobotFormProvider robot={{ type: activeType, name: 'Dual Arm', payload }}>
            <BimanualSO101FormFields />
            <Payload />
        </RobotFormProvider>,
        {
            route: `/projects/${PROJECT_ID}/robots/new`,
            path: '/projects/:project_id/robots/new',
        }
    );

const selectArm = async (user: ReturnType<typeof userEvent.setup>, arm: 'left' | 'right', name: string) => {
    await user.click(await screen.findByRole('button', { name: new RegExp(`Select ${arm} arm`) }));
    await user.click(await screen.findByRole('option', { name }));
    await user.keyboard('{Escape}');
};

describe('BimanualSO101FormFields', () => {
    it('renders a picker for both arms', async () => {
        server.use(
            http.get(ROBOTS_PATH, () =>
                HttpResponse.json([so101Robot('SO101_Follower', 'follower-a', 'Follower Arm A', 'SO101-A')])
            )
        );

        renderFields('BimanualSO101_Follower');

        expect(await screen.findByRole('button', { name: /Select left arm/ })).toBeVisible();
        expect(screen.getByRole('button', { name: /Select right arm/ })).toBeVisible();
    });

    it('only lists calibrated robots that have a serial number', async () => {
        server.use(
            http.get(ROBOTS_PATH, () =>
                HttpResponse.json([
                    so101Robot('SO101_Follower', 'follower-a', 'Follower Arm A', 'SO101-A'),
                    so101Robot('SO101_Follower', 'follower-b', 'No Calibration', 'SO101-B', false),
                    {
                        id: 'follower-c',
                        name: 'No Serial',
                        type: 'SO101_Follower' as const,
                        payload: {
                            connection_string: '',
                            serial_number: '',
                            calibration,
                        },
                    },
                ])
            )
        );
        const user = userEvent.setup();

        renderFields('BimanualSO101_Follower');
        await user.click(await screen.findByRole('button', { name: /Select left arm/ }));

        expect(await screen.findByRole('option', { name: 'Follower Arm A' })).toBeInTheDocument();
        expect(screen.queryByRole('option', { name: 'No Calibration' })).not.toBeInTheDocument();
        expect(screen.queryByRole('option', { name: 'No Serial' })).not.toBeInTheDocument();
    });

    it('copies serial number, calibration and role into the payload when selecting an arm', async () => {
        server.use(
            http.get(ROBOTS_PATH, () =>
                HttpResponse.json([
                    so101Robot('SO101_Follower', 'follower-a', 'Follower Arm A', 'SO101-A'),
                    so101Robot('SO101_Follower', 'follower-b', 'Follower Arm B', 'SO101-B'),
                ])
            )
        );
        const user = userEvent.setup();

        renderFields('BimanualSO101_Follower');
        await selectArm(user, 'left', 'Follower Arm A');

        expect(screen.getByRole('status')).toHaveTextContent(
            JSON.stringify({ left_serial_number: 'SO101-A', left_calibration: calibration, role: 'follower' })
        );
    });

    it('writes the leader role for a BimanualSO101_Leader robot', async () => {
        server.use(
            http.get(ROBOTS_PATH, () =>
                HttpResponse.json([
                    so101Robot('SO101_Leader', 'leader-a', 'Leader Arm A', 'SO101-A'),
                    so101Robot('SO101_Leader', 'leader-b', 'Leader Arm B', 'SO101-B'),
                ])
            )
        );
        const user = userEvent.setup();

        renderFields('BimanualSO101_Leader');
        await selectArm(user, 'right', 'Leader Arm B');

        expect(screen.getByRole('status')).toHaveTextContent(
            JSON.stringify({ right_serial_number: 'SO101-B', right_calibration: calibration, role: 'leader' })
        );
    });

    it('excludes the robot chosen for one arm from the other arm picker', async () => {
        server.use(
            http.get(ROBOTS_PATH, () =>
                HttpResponse.json([
                    so101Robot('SO101_Follower', 'follower-a', 'Follower Arm A', 'SO101-A'),
                    so101Robot('SO101_Follower', 'follower-b', 'Follower Arm B', 'SO101-B'),
                ])
            )
        );
        const user = userEvent.setup();

        renderFields('BimanualSO101_Follower');
        await selectArm(user, 'left', 'Follower Arm A');
        await user.click(screen.getByRole('button', { name: /Select right arm/ }));

        expect(await screen.findByRole('option', { name: 'Follower Arm B' })).toBeInTheDocument();
        expect(screen.queryByRole('option', { name: 'Follower Arm A' })).not.toBeInTheDocument();
    });

    it('preselects an arm from the existing payload', async () => {
        server.use(
            http.get(ROBOTS_PATH, () =>
                HttpResponse.json([
                    so101Robot('SO101_Follower', 'follower-a', 'Follower Arm A', 'SO101-A'),
                    so101Robot('SO101_Follower', 'follower-b', 'Follower Arm B', 'SO101-B'),
                ])
            )
        );

        renderFields('BimanualSO101_Follower', { left_serial_number: 'SO101-B' });

        expect(await screen.findByRole('button', { name: /Select left arm/ })).toHaveTextContent('Follower Arm B');
    });
});
