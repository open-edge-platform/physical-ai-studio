import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { CreateRobotForm } from './create-form';
import { RobotFormProvider } from './provider';

const definition = (type: string, calibration: { instructions: string; zero_tolerance_deg: number } | null) => ({
    type,
    display_name: type,
    role: 'follower' as const,
    urdf_path: `/api/robots/catalog/${type}/urdf`,
    package_map: {},
    joint_map: {},
    zero_calibration: calibration,
});

const renderCreateForm = (robotType: string) => {
    server.use(
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                definition('ReBot_B601_RS_Follower', { instructions: 'Fold the arm.', zero_tolerance_deg: 5 }),
                definition('Plain_Follower', null),
            ])
        ),
        http.get('/api/robots/catalog/{robot_type}/schema', () =>
            HttpResponse.json({ type: 'object', properties: {}, required: [] })
        )
    );

    return render(
        <RobotFormProvider robot={{ type: robotType, name: 'Khaos', payload: {} }}>
            <CreateRobotForm />
        </RobotFormProvider>,
        { route: '/projects/project-1/robots/new', path: '/projects/:project_id/robots/new' }
    );
};

describe('CreateRobotForm', () => {
    it('begins calibration for robot types that offer it', async () => {
        renderCreateForm('ReBot_B601_RS_Follower');

        expect(await screen.findByRole('button', { name: 'Begin Calibration' })).toBeInTheDocument();
    });

    it('adds robot types without calibration directly', async () => {
        renderCreateForm('Plain_Follower');

        expect(await screen.findByRole('button', { name: 'Add robot' })).toBeInTheDocument();
    });
});
