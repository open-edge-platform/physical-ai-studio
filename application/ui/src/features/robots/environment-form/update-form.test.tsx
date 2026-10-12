import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { render } from '../../../test-utils/render';
import { EnvironmentFormProvider } from './provider';
import { UpdateEnvironmentForm } from './update-form';

it('updates a named environment with empty robots and cameras', async () => {
    server.use(
        http.get('/api/projects/{project_id}/robots', () => HttpResponse.json([])),
        http.get('/api/projects/{project_id}/cameras', () => HttpResponse.json([])),
        http.get('/api/robots/catalog', () => HttpResponse.json([]))
    );

    const updateEnvironment = vi.fn();
    server.use(
        http.put('/api/projects/{project_id}/environments/{environment_id}', async ({ request }) => {
            updateEnvironment(await request.json());
            return HttpResponse.json(getMockedEnvironment());
        })
    );

    render(
        <EnvironmentFormProvider environment={{ name: 'My Environment', robots: [], cameras: [] }}>
            <UpdateEnvironmentForm />
        </EnvironmentFormProvider>,
        {
            route: '/projects/test-project-id/environments/environment-1/edit',
            path: '/projects/:project_id/environments/:environment_id/edit',
        }
    );

    const button = await screen.findByRole('button', { name: /update environment/i });
    expect(button).toBeEnabled();

    await userEvent.setup().click(button);

    await waitFor(() => {
        expect(updateEnvironment).toHaveBeenCalledWith({
            id: 'environment-1',
            name: 'My Environment',
            robots: [],
            cameras: [],
        });
    });
});
