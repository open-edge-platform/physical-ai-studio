import { ReactNode } from 'react';

import { ThemeProvider, ToastContainer } from '@geti-ui/ui';
import { QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouterProps, MemoryRouter as Router, RouterProvider } from 'react-router';

import { ZoomProvider } from './components/zoom/zoom';
import { queryClient } from './query-client/query-client';
import { router } from './router';

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <ZoomProvider>
                    <RouterProvider router={router} />
                    <ToastContainer />
                </ZoomProvider>
            </ThemeProvider>
        </QueryClientProvider>
    );
};

export const TestProviders = ({ children, routerProps }: { children: ReactNode; routerProps?: MemoryRouterProps }) => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <Router {...routerProps}>{children}</Router>
            </ThemeProvider>
        </QueryClientProvider>
    );
};
