// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { ErrorBoundary as ReactErrorBoundary } from 'react-error-boundary';

interface ErrorBoundaryProps {
    children: ReactNode;
    /** Rendered instead of `children` once an error has been caught. */
    fallback: (retry: () => void) => ReactNode;
}

/**
 * Contains render errors thrown by `children` instead of letting them bubble up.
 *
 * React Router's root `errorElement` catches uncaught render errors too, but it is
 * registered on the top-level route, so any error anywhere in the app unmounts the
 * *entire* routed tree and replaces it with a generic error page. Wrap any
 * self-contained, optional piece of UI (dialogs, panels, widgets that stream live
 * data) in this boundary instead, so a failure there can't take down the rest of
 * the application.
 *
 * A thin function-component wrapper around `react-error-boundary` (already a
 * dependency) rather than a hand-rolled class component, keeping the same
 * `fallback(retry)` render-prop shape its callers already use.
 */
export const ErrorBoundary = ({ children, fallback }: ErrorBoundaryProps) => (
    <ReactErrorBoundary
        onError={(error, info) => console.error('ErrorBoundary caught an error', error, info.componentStack)}
        fallbackRender={({ resetErrorBoundary }) => fallback(resetErrorBoundary)}
    >
        {children}
    </ReactErrorBoundary>
);
