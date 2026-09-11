// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Component, ReactNode } from 'react';

interface ErrorBoundaryProps {
    children: ReactNode;
    /** Rendered instead of `children` once an error has been caught. */
    fallback: (retry: () => void) => ReactNode;
}

interface ErrorBoundaryState {
    hasError: boolean;
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
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    state: ErrorBoundaryState = { hasError: false };

    static getDerivedStateFromError(): ErrorBoundaryState {
        return { hasError: true };
    }

    componentDidCatch(error: unknown, info: { componentStack?: string | null }) {
        console.error('ErrorBoundary caught an error', error, info.componentStack);
    }

    private retry = () => {
        this.setState({ hasError: false });
    };

    render() {
        if (this.state.hasError) {
            return this.props.fallback(this.retry);
        }

        return this.props.children;
    }
}
