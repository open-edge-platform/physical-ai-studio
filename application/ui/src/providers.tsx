import { ReactNode } from 'react';

import { Theme, useProvider } from '@adobe/react-spectrum';
import { ThemeProvider, ToastContainer } from '@geti-ui/ui';
import { QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouterProps, MemoryRouter as Router, RouterProvider } from 'react-router';

import { ZoomProvider } from './components/zoom/zoom';
import { queryClient } from './query-client/query-client';
import { router } from './router';

import theme from './custom-theme.module.css';

type CSSModule = Record<string, string>;

function mergeClasses(defaultObj: CSSModule = {}, customObj: CSSModule = {}): CSSModule {
    const merged = { ...defaultObj };
    for (const key in customObj) {
        if (merged[key]) {
            merged[key] = `${merged[key]} ${customObj[key]}`;
        } else {
            merged[key] = customObj[key];
        }
    }
    return merged;
}

const CustomThemeProvider = ({ children }: { children: ReactNode }) => {
    const { theme: defaultTheme } = useProvider();

    const getiTheme: Theme = {
        dark: mergeClasses(defaultTheme.dark, theme),
        light: mergeClasses(defaultTheme.light, theme),
        large: mergeClasses(defaultTheme.large, theme),
        medium: mergeClasses(defaultTheme.medium, theme),
        global: mergeClasses(defaultTheme.global, theme),
    };
    return <ThemeProvider theme={getiTheme}>{children}</ThemeProvider>;
};

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <CustomThemeProvider>
                    <ZoomProvider>
                        <RouterProvider router={router} />
                        <ToastContainer />
                    </ZoomProvider>
                </CustomThemeProvider>
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
