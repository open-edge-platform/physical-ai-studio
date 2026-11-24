// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { buildMetricSeries } from './metrics';
import { hasMetricValue, type MetricsEntry } from './types';

const entry = (values: Partial<MetricsEntry>): MetricsEntry => ({
    step: 0,
    epoch: null,
    train_loss: null,
    val_loss: null,
    ...values,
});

const learningRateSeries = (data: MetricsEntry[]) => {
    const series = buildMetricSeries(data).find((metric) => metric.title === 'Learning rate');
    if (!series?.lines) throw new Error('Expected learning-rate lines');
    return { ...series, lines: series.lines };
};

describe('learning-rate metrics', () => {
    it.each(['lr-AdamW', 'lr-AdamWBF16Kahan', 'lr-SGD'])('discovers %s without assuming an optimizer', (key) => {
        const row = entry({ [key]: 0.00001 });
        const series = learningRateSeries([row]);

        expect(series.lines).toHaveLength(1);
        expect(series.lines[0].name).toBe(key.slice(3));
        expect(series.lines[0].getY(row)).toBe(0.00001);
        expect(hasMetricValue(row, series.getY, series.lines)).toBe(true);
    });

    it('shows all parameter groups in a stable order and keeps sparse secondary-group points', () => {
        const data = [entry({ step: 49, 'lr-AdamW_pg2': 0.001 }), entry({ step: 99, 'lr-AdamW_pg1': 0.0001 })];
        const series = learningRateSeries(data);

        expect(series.lines.map((line) => line.name)).toEqual(['AdamW_pg1', 'AdamW_pg2']);
        expect(series.lines[0].color).not.toBe(series.lines[1].color);
        expect(data.filter((row) => hasMetricValue(row, series.getY, series.lines))).toHaveLength(2);
    });

    it('ignores empty learning-rate columns but includes zero rates', () => {
        const data = [entry({ 'lr-AdamW': null, 'lr-SGD': 0 })];
        const series = learningRateSeries(data);

        expect(series.lines.map((line) => line.name)).toEqual(['SGD']);
    });

    it('has no learning-rate data when no populated lr-* columns exist', () => {
        const row = entry({ train_loss: 0.5, 'lr-AdamW': null });
        const series = learningRateSeries([row]);

        expect(hasMetricValue(row, series.getY, series.lines)).toBe(false);
    });
});
