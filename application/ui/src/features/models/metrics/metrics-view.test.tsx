// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';

import { render } from '../../../test-utils/render';
import { MetricSeries, MetricsView } from './metrics';
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

    it('warns that validation metrics do not measure task success', () => {
        const data = [entry({ step: 1, val_loss: 0.5 })];

        render(<MetricsView series={[buildSeries('Validation loss', (e) => e.val_loss, data)]} isLoading={false} />);

        expect(screen.getByText('Validation metrics are a sanity check')).toBeInTheDocument();
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
