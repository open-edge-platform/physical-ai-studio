import { Suspense } from 'react';

import { ThemeProvider } from '@geti-ui/ui';
import { QueryClientProvider } from '@tanstack/react-query';
import { render as rtlRender, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { describe, expect, it } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { Robot } from '../../routes/robots/robot';
import { RobotDetailError } from './robot-detail-error';

const PROJECT_ID = 'project-id';
const ROBOT_ID = 'robot-id';

const renderAtShowRoute = () => {
    const queryClient = createQueryClient();
    const router = createMemoryRouter(
        [
            {
                path: '/projects/:project_id/robots/:robot_id',
                element: <Robot />,
                errorElement: <RobotDetailError />,
            },
            {
                path: '/projects/:project_id/robots',
                element: <div>Robots index</div>,
            },
        ],
        { initialEntries: [`/projects/${PROJECT_ID}/robots/${ROBOT_ID}`], initialIndex: 0 }
    );

    return rtlRender(
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <Suspense>
                    <RouterProvider router={router} />
                </Suspense>
            </ThemeProvider>
        </QueryClientProvider>
    );
};

describe('RobotDetailError', () => {
    it('shows a friendly message when the robot is not found', async () => {
        server.use(
            http.get('/api/projects/{project_id}/robots/{robot_id}', () =>
                HttpResponse.json(
                    { error_code: 'robot_not_found', message: 'Robot not found', http_status: 404 } as never,
                    { status: 404 }
                )
            )
        );

        renderAtShowRoute();

        expect(await screen.findByText('This robot no longer exists.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Back to robots' })).toBeInTheDocument();
    });

    it('navigates back to the robots index', async () => {
        server.use(
            http.get('/api/projects/{project_id}/robots/{robot_id}', () =>
                HttpResponse.json(
                    { error_code: 'robot_not_found', message: 'Robot not found', http_status: 404 } as never,
                    { status: 404 }
                )
            )
        );
        const user = userEvent.setup();

        renderAtShowRoute();

        await user.click(await screen.findByRole('button', { name: 'Back to robots' }));

        expect(await screen.findByText('Robots index')).toBeInTheDocument();
    });
});
