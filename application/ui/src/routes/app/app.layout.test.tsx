import { ThemeProvider } from '@geti-ui/ui';
import { QueryClientProvider } from '@tanstack/react-query';
import { render as rtlRender, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { vi } from 'vitest';

import { RestartStateProvider } from '../../features/system/restart-state';
import { createQueryClient } from '../../query-client/query-client';
import { render } from '../../test-utils/render';
import { AppLayout } from './app.layout';

vi.mock('../../features/jobs/footer/job-status', () => ({
    JobStatus: () => null,
}));

describe('AppLayout', () => {
    it('renders the logo, linking to the projects page', () => {
        render(<AppLayout />, { route: '/projects', path: '/projects' });

        const logoLink = screen.getByRole('link', { name: /physical ai studio/i });

        expect(logoLink).toBeInTheDocument();
        expect(logoLink).toHaveAttribute('href', '/projects');
    });

    it('renders the child route content in the content area', () => {
        const queryClient = createQueryClient();
        const router = createMemoryRouter(
            [
                {
                    path: '/',
                    element: (
                        <QueryClientProvider client={queryClient}>
                            <ThemeProvider>
                                <RestartStateProvider>
                                    <AppLayout />
                                </RestartStateProvider>
                            </ThemeProvider>
                        </QueryClientProvider>
                    ),
                    children: [{ path: 'projects', element: <div>child content</div> }],
                },
            ],
            { initialEntries: ['/projects'] }
        );

        rtlRender(<RouterProvider router={router} />);

        expect(screen.getByText('child content')).toBeInTheDocument();
    });
});
