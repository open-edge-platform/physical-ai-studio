// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { EnvironmentFormProvider } from './provider';
import { AddRobotForm } from './robot-form';

const PROJECT_ID = 'project-1';

const definition = (type: string, role: 'follower' | 'leader', simulated: boolean) => ({
    type,
    display_name: type,
    role,
    urdf_path: '',
    package_map: {},
    joint_map: {},
    simulation: simulated
        ? { scenes: [{ id: 'conveyor', display_name: 'Conveyor', description: '' }], default_scene: 'conveyor' }
        : null,
});

const robot = (id: string, name: string, type: string) => ({ id, name, type, payload: {} });

const renderForm = (onAddRobot = rs.fn()) => {
    server.use(
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                definition('MuJoCo_SO101_Follower', 'follower', true),
                definition('SO101_Follower', 'follower', false),
                definition('SO101_Leader', 'leader', false),
            ] as never)
        ),
        http.get('/api/projects/{project_id}/robots', () =>
            HttpResponse.json([
                robot('sim', 'Sim arm', 'MuJoCo_SO101_Follower'),
                robot('arm', 'Real arm', 'SO101_Follower'),
                robot('leader', 'Leader arm', 'SO101_Leader'),
            ] as never)
        )
    );
    render(
        <EnvironmentFormProvider>
            <AddRobotForm onAddRobot={onAddRobot} />
        </EnvironmentFormProvider>,
        { route: `/projects/${PROJECT_ID}/environments/new`, path: '/projects/:project_id/environments/new' }
    );
    return onAddRobot;
};

const pick = async (picker: RegExp, option: string) => {
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: picker }));
    await user.click(await screen.findByRole('option', { name: option }));
};

describe('AddRobotForm', () => {
    it('lets the autopilot lead a simulated follower', async () => {
        const onAddRobot = renderForm();

        await pick(/Robot \(Follower\)/, 'Sim arm');
        await pick(/Robot \(Leader, optional\)/, 'Autopilot (simulation)');
        await userEvent.setup().click(screen.getByRole('button', { name: 'Add' }));

        expect(onAddRobot).toHaveBeenCalledWith({ robot_id: 'sim', teleoperator: { type: 'autopilot' } });
    });

    it('offers no autopilot for a follower without a simulation', async () => {
        renderForm();

        await pick(/Robot \(Follower\)/, 'Real arm');
        await userEvent.setup().click(screen.getByRole('button', { name: /Robot \(Leader, optional\)/ }));

        expect(await screen.findByRole('option', { name: 'Leader arm' })).toBeInTheDocument();
        expect(screen.queryByRole('option', { name: 'Autopilot (simulation)' })).not.toBeInTheDocument();
    });

    it('drops the autopilot when the follower changes to one without a simulation', async () => {
        const onAddRobot = renderForm();

        await pick(/Robot \(Follower\)/, 'Sim arm');
        await pick(/Robot \(Leader, optional\)/, 'Autopilot (simulation)');
        await pick(/Robot \(Follower\)/, 'Real arm');
        await userEvent.setup().click(screen.getByRole('button', { name: 'Add' }));

        expect(onAddRobot).toHaveBeenCalledWith({ robot_id: 'arm', teleoperator: { type: 'none' } });
    });
});
