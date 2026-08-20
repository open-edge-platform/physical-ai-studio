import { useId, useMemo } from 'react';

import { Grid, Heading, IllustratedMessage, Loading } from '@geti-ui/ui';
import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { fetchSSE } from '../../../api/fetch-sse';
import { getQueryKey } from '../../../query-client/query-client.interface';
import { ReactComponent as EmptyIllustration } from './../../../assets/illustration.svg';
import { mergeMetricsByStep } from './merge-metrics-by-step';
import { MetricGraph } from './metric-graph';
import { MetricsEntry } from './types';

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
    data: MetricsEntry[];
    color?: string;
    getX: (metricsEntry: MetricsEntry) => number;
    getY: (metricsEntry: MetricsEntry) => number | null | undefined;
}

interface MetricsViewProps {
    series: MetricSeries[];
    isLoading: boolean;
}

const MetricsView = ({ series, isLoading }: MetricsViewProps) => {
    const syncId = useId();

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
            {seriesReadyToRender.map(({ title, xLabel, yLabel, data, color, getX, getY }) => (
                <MetricGraph
                    key={title}
                    syncId={syncId}
                    title={title}
                    yAxisLabel={yLabel}
                    xAxisLabel={xLabel}
                    data={data}
                    color={color}
                    getX={getX}
                    getY={getY}
                />
            ))}
        </Grid>
    );
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
        select: mergeMetricsByStep,
        staleTime: Infinity,
    });
};

export const JobMetricsContent = ({ jobId }: { jobId: string }) => {
    const query = useJobMetrics(jobId);
    const metricsData = useMemo(() => {
        return query.data ?? [];
    }, [query.data]);

    const metrics: MetricSeries[] = [
        {
            title: 'Training loss',
            xLabel: 'Step',
            yLabel: 'Loss',
            data: metricsData,
            getX: (entry) => entry.step,
            getY: (entry) => entry.train_loss,
            color: 'var(--moss-tint-1)'
        },
        {
            title: 'Validation loss',
            xLabel: 'Step',
            yLabel: 'Loss',
            data: metricsData,
            getX: (entry) => entry.step,
            getY: (entry) => entry.val_loss,
            color: 'var(--coral)',
        },
        {
            title: 'Learning rate',
            xLabel: 'Step',
            yLabel: 'Learning rate',
            data: metricsData,
            getX: (entry) => entry.step,
            getY: (entry) => entry['lr-AdamW'],
        },
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
        select: mergeMetricsByStep,
        staleTime: Infinity,
    });
};

export const MetricsContent = ({ modelId }: { modelId: string }) => {
    const query = useModelMetrics(modelId);

    const metricsData = useMemo(() => {
        return query.data ?? [];
    }, [query.data]);

    const metrics: MetricSeries[] = [
        {
            title: 'Training loss',
            xLabel: 'Step',
            yLabel: 'Loss',
            data: metricsData,
            color: 'var(--moss-tint-1)',
            getX: (entry) => entry.step,
            getY: (entry) => entry.train_loss,
        },
        {
            title: 'Validation loss',
            xLabel: 'Step',
            yLabel: 'Loss',
            data: metricsData,
            color: 'var(--coral)',
            getX: (entry) => entry.step,
            getY: (entry) => entry.val_loss,
        },
        {
            title: 'Learning rate',
            xLabel: 'Step',
            yLabel: 'Learning rate',
            data: metricsData,
            getX: (entry) => entry.step,
            getY: (entry) => entry['lr-AdamW'],
        },
    ];

    return <MetricsView series={metrics} isLoading={query.isLoading} />;
};
