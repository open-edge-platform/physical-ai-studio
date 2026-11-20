// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { render } from '../../../test-utils/render';
import { buildMetricSeries, MetricSeries, MetricsView } from './metrics';
import type { MetricsEntry } from './types';

const entry = (overrides: Partial<MetricsEntry>): MetricsEntry => ({
    step: 0,
    epoch: null,
    train_loss: null,
    val_loss: null,
    'lr-AdamW': null,
    ...overrides,
});

const buildSeries = (title: string, getY: MetricSeries['getY'], data: MetricsEntry[]): MetricSeries => ({
    title,
    xLabel: 'Step',
    yLabel: 'Loss',
    data,
    getX: (metricsEntry) => metricsEntry.step,
    getY,
});

describe('MetricsView', () => {
    it('shows a loading indicator while loading, regardless of series content', () => {
        render(<MetricsView series={[]} isLoading />);

        expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    it('shows the empty state when no series has any data points', () => {
        render(<MetricsView series={[]} isLoading={false} />);

        expect(screen.getByText('No metrics available yet')).toBeInTheDocument();
    });

    it('shows the empty state when every series has entries but all values are null', () => {
        const data = [entry({ step: 1 }), entry({ step: 2 })];
        const series = [buildSeries('Training loss', (e) => e.train_loss, data)];

        render(<MetricsView series={series} isLoading={false} />);

        expect(screen.getByText('No metrics available yet')).toBeInTheDocument();
    });

    it('renders a chart for a series that has at least one non-null value', () => {
        const data = [entry({ step: 1, train_loss: 0.5 })];
        const series = [buildSeries('Training loss', (e) => e.train_loss, data)];

        render(<MetricsView series={series} isLoading={false} />);

        expect(screen.getByText('Training loss')).toBeInTheDocument();
    });

    it.each([
        ['Training loss', /optimization loss on training demonstrations/],
        ['Validation loss', /Loss on held-out demonstrations, not task success/],
        ['Validation action error', /Mean absolute difference between predicted and recorded actions/],
        ['Learning rate', /Separate lines represent optimizers or parameter groups/],
    ])('shows focused help for %s only when requested', async (title, description) => {
        const user = userEvent.setup();
        const data = [entry({ train_loss: 0.5, val_loss: 0.6, val_action_error: 4, 'lr-AdamW': 0.0001 })];
        render(<MetricsView series={buildMetricSeries(data)} isLoading={false} />);

        expect(screen.queryByText('Validation metrics are a sanity check')).not.toBeInTheDocument();
        expect(screen.queryByText(description)).not.toBeInTheDocument();
        const helpButton = screen.getByRole('button', { name: `${title} information` });
        expect(helpButton.closest('h5')).toHaveTextContent(title);
        await user.click(helpButton);

        const dialog = await screen.findByRole('dialog');
        expect(within(dialog).getByRole('heading', { name: title })).toBeVisible();
        expect(within(dialog).getByText(description)).toBeVisible();
        await user.click(within(dialog).getByRole('button', { name: 'Dismiss' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        await waitFor(() => expect(helpButton).toHaveFocus());
    });

    it('hides a chart whose values are all null while still rendering charts that have data', () => {
        const data = [entry({ step: 1, train_loss: 0.5, val_loss: null })];
        const series = [
            buildSeries('Training loss', (e) => e.train_loss, data),
            buildSeries('Validation loss', (e) => e.val_loss, data),
        ];

        render(<MetricsView series={series} isLoading={false} />);

        expect(screen.getByText('Training loss')).toBeInTheDocument();
        expect(screen.queryByText('Validation loss')).not.toBeInTheDocument();
    });

    it('renders a multi-line chart when only a secondary line has data', () => {
        const data = [entry({ 'lr-AdamW_pg2': 0.001 })];
        const series = buildSeries('Learning rate', () => null, data);
        series.lines = [{ name: 'AdamW_pg2', color: 'blue', getY: (row) => row['lr-AdamW_pg2'] }];

        render(<MetricsView series={[series]} isLoading={false} />);

        expect(screen.getByText('Learning rate')).toBeInTheDocument();
        expect(screen.queryByText('No metrics available yet')).not.toBeInTheDocument();
    });
});
