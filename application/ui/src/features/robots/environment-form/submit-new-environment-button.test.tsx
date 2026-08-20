import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { EnvironmentForm, EnvironmentFormProvider } from './provider';
import { SubmitNewEnvironmentButton } from './submit-new-environment-button';

const PROJECT_ID = 'test-project-id';

const ROBOTS_PATH = '/api/projects/{project_id}/robots';

const renderButton = (environment: Partial<EnvironmentForm> = {}) => {
    return render(
        <EnvironmentFormProvider environment={{ name: '', robots: [], cameras: [], ...environment }}>
            <SubmitNewEnvironmentButton />
        </EnvironmentFormProvider>,
        {
            route: `/projects/${PROJECT_ID}/environments/new`,
            path: '/projects/:project_id/environments/new',
        }
    );
};

describe('SubmitNewEnvironmentButton', () => {
    describe('is disabled', () => {
        it('when the environment name is empty', async () => {
            server.use(http.get(ROBOTS_PATH, () => HttpResponse.json([])));

            renderButton({ name: '' });

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

            renderButton({ name: 'My Environment', robots: [] });

            expect(await screen.findByRole('button', { name: /add environment/i })).toBeDisabled();
        });
    });

    describe('is enabled', () => {
        it('when the name is set and the project has no robots', async () => {
            server.use(http.get(ROBOTS_PATH, () => HttpResponse.json([])));

            renderButton({ name: 'My Environment' });

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

            renderButton({
                name: 'My Environment',
                robots: [{ robot_id: 'robot-1', teleoperator: { type: 'none' } }],
            });

            expect(await screen.findByRole('button', { name: /add environment/i })).not.toBeDisabled();
        });
    });
});
