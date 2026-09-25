// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { render } from '../../test-utils/render';
import { ErrorBoundary } from './error-boundary';

const Boom = () => {
    throw new Error('boom');
};

describe('ErrorBoundary', () => {
    it('renders children when nothing throws', () => {
        render(
            <ErrorBoundary fallback={() => <div>fallback</div>}>
                <div>fine</div>
            </ErrorBoundary>
        );

        expect(screen.getByText('fine')).toBeInTheDocument();
        expect(screen.queryByText('fallback')).not.toBeInTheDocument();
    });

    it('renders the fallback once a child throws while rendering', () => {
        // Errors thrown while rendering are expected here; silence React's console.error noise for this test.
        vi.spyOn(console, 'error').mockImplementation(() => undefined);

        render(
            <ErrorBoundary fallback={() => <div>fallback</div>}>
                <Boom />
            </ErrorBoundary>
        );

        expect(screen.getByText('fallback')).toBeInTheDocument();
    });

    it('re-renders children once retry is called', async () => {
        vi.spyOn(console, 'error').mockImplementation(() => undefined);
        const user = userEvent.setup();
        let shouldThrow = true;
        const Flaky = () => {
            if (shouldThrow) {
                throw new Error('boom');
            }
            return <div>recovered</div>;
        };

        render(
            <ErrorBoundary fallback={(retry) => <button onClick={retry}>retry</button>}>
                <Flaky />
            </ErrorBoundary>
        );

        expect(screen.getByRole('button', { name: 'retry' })).toBeInTheDocument();
        shouldThrow = false;
        await user.click(screen.getByRole('button', { name: 'retry' }));

        expect(await screen.findByText('recovered')).toBeInTheDocument();
    });
});
