import { createContext, Suspense, useContext, type ReactNode } from 'react';

import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
    render as rtlRender,
    renderHook as rtlRenderHook,
    type RenderOptions as RTLRenderOptions,
} from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { createQueryClient } from '../query-client/query-client';

type RenderOptions = RTLRenderOptions & {
    /** The URL the memory router starts at, e.g. '/projects/abc/environments/new'. */
    route?: string;
    /** The route pattern that matches `route`, e.g. '/projects/:project_id/environments/new'. */
    path?: string;
    /** Pass an existing QueryClient to share cache across multiple render calls in one test. */
    queryClient?: QueryClient;
};

const TestProviders = ({ children, queryClient }: { children: ReactNode; queryClient: QueryClient }) => (
    <QueryClientProvider client={queryClient}>
        <ThemeProvider>
            <Suspense>{children}</Suspense>
        </ThemeProvider>
    </QueryClientProvider>
);

const createTestRouter = (
    children: ReactNode,
    options: Pick<RenderOptions, 'route' | 'path'>,
    queryClient: QueryClient
) => {
    const route = options.route ?? '/';
    const path = options.path ?? '/';

    return createMemoryRouter(
        [
            {
                path,
                element: <TestProviders queryClient={queryClient}>{children}</TestProviders>,
            },
        ],
        { initialEntries: [route], initialIndex: 0 }
    );
};

export const render = (ui: ReactNode, options: RenderOptions = {}) => {
    const { route, path, queryClient: queryClientOption, wrapper: _wrapper, ...rtlOptions } = options;
    const testQueryClient = queryClientOption ?? createQueryClient();
    const router = createTestRouter(ui, { route, path }, testQueryClient);

    return rtlRender(<RouterProvider router={router} />, rtlOptions);
};

// Context used by renderHook to thread dynamic children through a stable router instance.
// The router is created once per renderHook call; children update via this context on rerender.
const HookChildrenContext = createContext<ReactNode>(null);

const HookRouteElement = ({ queryClient }: { queryClient: QueryClient }) => {
    const children = useContext(HookChildrenContext);
    return <TestProviders queryClient={queryClient}>{children}</TestProviders>;
};

export const renderHook = <TProps, TResult>(callback: (props: TProps) => TResult, options: RenderOptions = {}) => {
    const { route, path, queryClient: queryClientOption } = options;
    const testQueryClient = queryClientOption ?? createQueryClient();

    // Create the router once so that rerenders don't reset router state or remount the hook.
    const router = createMemoryRouter(
        [{ path: path ?? '/', element: <HookRouteElement queryClient={testQueryClient} /> }],
        { initialEntries: [route ?? '/'], initialIndex: 0 }
    );

    const Wrapper = ({ children }: { children: ReactNode }) => {
        const wrappedChildren = options.wrapper ? <options.wrapper>{children}</options.wrapper> : children;

        return (
            <HookChildrenContext.Provider value={wrappedChildren}>
                <RouterProvider router={router} />
            </HookChildrenContext.Provider>
        );
    };

    return rtlRenderHook(callback, { wrapper: Wrapper });
};
