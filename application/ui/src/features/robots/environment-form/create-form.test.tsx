import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { CreateEnvironmentForm } from './create-form';
import { EnvironmentForm, EnvironmentFormProvider } from './provider';

const PROJECT_ID = 'test-project-id';

const ROBOTS_PATH = '/api/projects/{project_id}/robots';
const CAMERAS_PATH = '/api/projects/{project_id}/cameras';
const ENVIRONMENTS_PATH = '/api/projects/{project_id}/environments';
const CATALOG_PATH = '/api/robots/catalog';

const so101FollowerDefinition = {
    type: 'SO101_Follower',
    display_name: 'SO101 Follower',
    role: 'follower',
    urdf_path: '/api/robots/catalog/SO101_Follower/urdf',
    package_map: {},
    joint_map: {},
} as const;

const renderForm = (environment: Partial<EnvironmentForm> = {}) => {
    return render(
        <EnvironmentFormProvider environment={{ name: '', robots: [], cameras: [], ...environment }}>
            <CreateEnvironmentForm />
        </EnvironmentFormProvider>,
        {
            route: `/projects/${PROJECT_ID}/environments/new`,
            path: '/projects/:project_id/environments/new',
        }
    );
};

describe('CreateEnvironmentForm', () => {
    beforeEach(() => {
        server.use(http.get(CAMERAS_PATH, () => HttpResponse.json([])));
        server.use(http.get(CATALOG_PATH, () => HttpResponse.json([so101FollowerDefinition])));
    });

    describe('is disabled', () => {
        it('when the environment name is empty', async () => {
            server.use(http.get(ROBOTS_PATH, () => HttpResponse.json([])));

            renderForm({ name: '' });

            expect(await screen.findByRole('button', { name: /add environment/i })).toBeDisabled();
        });

        it('when the project has robots but none were added to the environment', async () => {
            server.use(
                http.get(ROBOTS_PATH, () =>
                    HttpResponse.json([
                        {
                            id: 'robot-1',
                            type: 'SO101_Follower',
                            name: 'Test Robot',
                            payload: { connection_string: '', serial_number: '' },
                        },
                    ])
                )
            );

            renderForm({ name: 'My Environment', robots: [] });

            expect(await screen.findByRole('button', { name: /add environment/i })).toBeDisabled();
        });
    });

    describe('is enabled', () => {
        it('when the name is set and the project has no robots', async () => {
            server.use(http.get(ROBOTS_PATH, () => HttpResponse.json([])));

            renderForm({ name: 'My Environment' });

            expect(await screen.findByRole('button', { name: /add environment/i })).not.toBeDisabled();
        });

        it('when the name is set and at least one robot was added to the environment', async () => {
            server.use(
                http.get(ROBOTS_PATH, () =>
                    HttpResponse.json([
                        {
                            id: 'robot-1',
                            type: 'SO101_Follower',
                            name: 'Test Robot',
                            payload: { connection_string: '', serial_number: '' },
                        },
                    ])
                )
            );

            renderForm({
                name: 'My Environment',
                robots: [{ robot_id: 'robot-1', teleoperator: { type: 'none' } }],
            });

            expect(await screen.findByRole('button', { name: /add environment/i })).not.toBeDisabled();
        });
    });

    it('submits exactly once when pressing Enter in the name field', async () => {
        server.use(http.get(ROBOTS_PATH, () => HttpResponse.json([])));
        const createEnvironmentSpy = vi.fn();

        server.use(
            http.post(ENVIRONMENTS_PATH, async ({ request }) => {
                createEnvironmentSpy();
                const body = await request.json();

                return HttpResponse.json({ ...body, id: 'created-environment-id' });
            })
        );

        const user = userEvent.setup();

        renderForm();

        const nameField = await screen.findByLabelText(/name/i);

        await user.type(nameField, 'My Environment{Enter}');

        expect(createEnvironmentSpy).toHaveBeenCalledOnce();
    });
});
