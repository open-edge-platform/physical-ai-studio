import { useMemo } from 'react';

import { Grid, Heading, IllustratedMessage, Loading } from '@geti-ui/ui';
import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { fetchSSE } from '../../../api/fetch-sse';
import { getQueryKey } from '../../../query-client/query-client.interface';
import { ReactComponent as EmptyIllustration } from './../../../assets/illustration.svg';
import { MetricGraph, type MetricGraphPoint } from './metric-graph';

const NoMetricsAvailable = () => {
    return (
        <IllustratedMessage marginY='size-400'>
            <EmptyIllustration height='250px' />
            <Heading>No metrics available yet</Heading>
        </IllustratedMessage>
    );
};

interface MetricSeries {
    title: string;
    xLabel: string;
    yLabel: string;
    data: MetricGraphPoint[];
    color?: string;
}

interface MetricsViewProps {
    series: MetricSeries[];
    isLoading: boolean;
}

const MetricsView = ({ series, isLoading }: MetricsViewProps) => {
    if (isLoading) {
        return <Loading mode='inline' />;
    }

    const seriesReadyToRender = series.filter((metric) => metric.data.length > 0);

    if (seriesReadyToRender.length === 0) {
        return <NoMetricsAvailable />;
    }

    return (
        <Grid
            columns='repeat(auto-fit, minmax(min(100%, var(--spectrum-global-dimension-size-6000)), 1fr))'
            gap='size-200'
        >
            {seriesReadyToRender.map(({ title, xLabel, yLabel, data, color }) => (
                <MetricGraph
                    key={title}
                    title={title}
                    yAxisLabel={yLabel}
                    xAxisLabel={xLabel}
                    data={data}
                    color={color}
                />
            ))}
        </Grid>
    );
};

interface MetricsEntry {
    epoch: number | null;
    step: number;
    train_loss: number | null | undefined;
    train_loss_step: number | null | undefined;
    'lr-AdamW': number | null | undefined;
    val_loss: number | null | undefined;
}

const selectSeries = (
    data: MetricsEntry[] | undefined,
    getX: (metricsEntry: MetricsEntry) => number,
    getY: (metricsEntry: MetricsEntry) => number | null | undefined
) => {
    if (data == null) return [];

    return data.flatMap((entry) => {
        const y = getY(entry);
        return y == null ? [] : [{ x: getX(entry), y }];
    });
};

const useJobMetrics = (jobId: string) => {
    return useQuery({
        queryKey: getQueryKey(['get', '/api/jobs/{job_id}/model_metrics', { params: { path: { job_id: jobId } } }]),
        queryFn: streamedQuery({
            streamFn: (context) => {
                const url = fetchClient.PATH('/api/jobs/{job_id}/model_metrics', {
                    params: { path: { job_id: jobId } },
                });

                return fetchSSE<MetricsEntry>(url, { signal: context.signal });
            },
        }),
        staleTime: Infinity,
    });
};

export const JobMetricsContent = ({ jobId }: { jobId: string }) => {
    const query = useJobMetrics(jobId);

    const lossStepMetrics = useMemo(() => {
        return selectSeries(
            query.data,
            (entry) => entry.step,
            (entry) => entry.train_loss ?? entry.train_loss_step
        );
    }, [query.data]);

    const validationLossStepMetrics = useMemo(() => {
        return selectSeries(
            query.data,
            (entry) => entry.step,
            (entry) => entry.val_loss
        );
    }, [query.data]);

    const learningRateStepMetrics = useMemo(() => {
        return selectSeries(
            query.data,
            (entry) => entry.step,
            (entry) => entry['lr-AdamW']
        );
    }, [query.data]);

    const metrics = [
        { title: 'Training loss', xLabel: 'Step', yLabel: 'Loss', data: lossStepMetrics, color: 'var(--moss-tint-1)' },
        {
            title: 'Validation loss',
            xLabel: 'Step',
            yLabel: 'Loss',
            data: validationLossStepMetrics,
            color: 'var(--coral)',
        },
        { title: 'Learning rate', xLabel: 'Step', yLabel: 'Learning rate', data: learningRateStepMetrics },
    ];

    return <MetricsView series={metrics} isLoading={query.isLoading} />;
};

const useModelMetrics = (modelId: string) => {
    return useQuery({
        queryKey: getQueryKey(['get', '/api/models/{model_id}/metrics', { params: { path: { model_id: modelId } } }]),
        queryFn: streamedQuery({
            streamFn: (context) => {
                const url = fetchClient.PATH('/api/models/{model_id}/metrics', {
                    params: { path: { model_id: modelId } },
                });

                return fetchSSE<MetricsEntry>(url, { signal: context.signal });
            },
        }),
        staleTime: Infinity,
    });
};

export const MetricsContent = ({ modelId }: { modelId: string }) => {
    const query = useModelMetrics(modelId);

    const lossStepMetrics = useMemo(() => {
        return selectSeries(
            query.data,
            (entry) => entry.step,
            (entry) => entry.train_loss ?? entry.train_loss_step
        );
    }, [query.data]);

    const validationLossStepMetrics = useMemo(() => {
        return selectSeries(
            query.data,
            (entry) => entry.step,
            (entry) => entry.val_loss
        );
    }, [query.data]);

    const learningRateStepMetrics = useMemo(() => {
        return selectSeries(
            query.data,
            (entry) => entry.step,
            (entry) => entry['lr-AdamW']
        );
    }, [query.data]);

    const metrics = [
        { title: 'Training loss', xLabel: 'Step', yLabel: 'Loss', data: lossStepMetrics, color: 'var(--moss-tint-1)' },
        {
            title: 'Validation loss',
            xLabel: 'Step',
            yLabel: 'Loss',
            data: validationLossStepMetrics,
            color: 'var(--coral)',
        },
        { title: 'Learning rate', xLabel: 'Step', yLabel: 'Learning rate', data: learningRateStepMetrics },
    ];

    return <MetricsView series={metrics} isLoading={query.isLoading} />;
};
