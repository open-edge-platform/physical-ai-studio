import { useId, useMemo, useState } from 'react';

import { Content, Flex, Grid, Heading, IllustratedMessage, InlineAlert, Loading, StatusLight } from '@geti-ui/ui';
import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { batchAsyncIterable } from '../../../api/batch-async-iterable';
import { fetchClient } from '../../../api/client';
import { fetchSSE } from '../../../api/fetch-sse';
import { SchemaOperationalIssue } from '../../../api/openapi-spec';
import { issueToasts } from '../../../components/notification/issue-toasts';
import { getQueryKey } from '../../../query-client/query-client.interface';
import { ReactComponent as EmptyIllustration } from './../../../assets/illustration.svg';
import { foldMetricsBatch, sortMetricsByStep } from './merge-metrics-by-step';
import { MetricGraph } from './metric-graph';
import type { MetricLine, MetricsEntry } from './types';

const NoMetricsAvailable = () => {
    return (
        <IllustratedMessage marginY='size-400'>
            <EmptyIllustration height='250px' />
            <Heading>No metrics available yet</Heading>
        </IllustratedMessage>
    );
};

const ValidationMetricsNote = () => (
    <InlineAlert variant='info'>
        <Heading>Validation metrics are a sanity check</Heading>
        <Content>
            They don&apos;t tell you how well the robot will do the task. Validation loss can go up while the policy
            keeps improving, especially for flow-matching policies such as SmolVLA and Pi0.5. Action error is the
            average gap between predicted and recorded actions, in the units the dataset stores (for SO-101: -100 to 100
            per joint, 0 to 100 for the gripper). Test the model on your robot or in simulation to judge it.
        </Content>
    </InlineAlert>
);

export interface MetricSeries {
    title: string;
    xLabel: string;
    yLabel: string;
    data: MetricsEntry[];
    color?: string;
    getX: (metricsEntry: MetricsEntry) => number;
    getY: (metricsEntry: MetricsEntry) => number | null | undefined;
    lines?: MetricLine[];
    formatY?: (y: number) => string;
}

interface MetricsViewProps {
    series: MetricSeries[];
    isLoading: boolean;
}

export const MetricsView = ({ series, isLoading }: MetricsViewProps) => {
    const syncId = useId();

    if (isLoading) {
        return <Loading mode='inline' />;
    }

    const seriesReadyToRender = series.filter((metric) => metric.data.some((entry) => metric.getY(entry) != null));

    if (seriesReadyToRender.length === 0) {
        return <NoMetricsAvailable />;
    }

    return (
        <Flex direction='column' gap='size-200'>
            <ValidationMetricsNote />
            <Grid
                columns='repeat(auto-fit, minmax(min(100%, var(--spectrum-global-dimension-size-6000)), 1fr))'
                gap='size-200'
            >
                {seriesReadyToRender.map(({ title, xLabel, yLabel, data, color, getX, getY, lines, formatY }) => (
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
                        lines={lines}
                        formatY={formatY}
                    />
                ))}
            </Grid>
        </Flex>
    );
};

// Flush interval for batching incoming SSE metric rows into a single query-data
// update, so the UI re-renders/re-merges on a bounded cadence (regardless of how
// many rows a job/model produced) instead of once per raw row.
const METRICS_BATCH_INTERVAL_MS = 200;

const useJobMetrics = (jobId: string) => {
    const [issues, setIssues] = useState<SchemaOperationalIssue[]>([]);
    const query = useQuery({
        queryKey: getQueryKey(['get', '/api/jobs/{job_id}/model_metrics', { params: { path: { job_id: jobId } } }]),
        queryFn: streamedQuery({
            streamFn: (context) => {
                const url = fetchClient.PATH('/api/jobs/{job_id}/model_metrics', {
                    params: { path: { job_id: jobId } },
                });

                return batchAsyncIterable(
                    fetchSSE<MetricsEntry>(url, {
                        signal: context.signal,
                        onIssues: (data) => {
                            if (context.signal.aborted) return;
                            const snapshot: SchemaOperationalIssue[] = JSON.parse(data);
                            setIssues(snapshot);
                            issueToasts.replaceSource(`metrics:${jobId}`, snapshot);
                        },
                    }),
                    METRICS_BATCH_INTERVAL_MS
                );
            },
            reducer: foldMetricsBatch,
            initialValue: new Map<number, MetricsEntry>(),
        }),
        select: sortMetricsByStep,
        staleTime: 0,
        // Retry ended/empty streams without interrupting an active SSE connection.
        refetchInterval: ({ state }) => (state.fetchStatus === 'fetching' ? false : 5000),
    });
    return { ...query, issues };
};

const lossFormatter = new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 4,
    maximumSignificantDigits: 4,
    roundingPriority: 'morePrecision',
});

const learningRateFormatter = new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 6,
    maximumSignificantDigits: 4,
    roundingPriority: 'morePrecision',
});

const ACTION_ERROR_PREFIX = 'val_action_error_';
const JOINT_COLORS = [
    'orange',
    'magenta',
    'seafoam',
    'yellow',
    'purple',
    'red',
    'celery',
    'indigo',
    'fuchsia',
    'green',
].map((color) => `var(--spectrum-global-color-${color}-500)`);

// One line per joint (the `val/action_error/<joint>` columns), plus the mean across joints.
const actionErrorLines = (data: MetricsEntry[]): MetricLine[] => {
    const keys = new Set(data.flatMap((entry) => Object.keys(entry).filter((k) => k.startsWith(ACTION_ERROR_PREFIX))));
    return [
        {
            name: 'Mean',
            color: 'var(--energy-blue)',
            strokeWidth: 4,
            strokeDasharray: '8 5',
            getY: (entry) => entry.val_action_error,
        },
        ...[...keys].map((key, index) => ({
            name: key.slice(ACTION_ERROR_PREFIX.length).replace(/\.pos$/, ''),
            color: JOINT_COLORS[index % JOINT_COLORS.length],
            getY: (entry: MetricsEntry) => entry[key],
        })),
    ];
};

const buildMetricSeries = (data: MetricsEntry[]): MetricSeries[] => [
    {
        title: 'Training loss',
        xLabel: 'Step',
        yLabel: 'Loss',
        data,
        color: 'var(--moss-tint-1)',
        getX: (entry) => entry.step,
        getY: (entry) => entry.train_loss,
        formatY: (y) => lossFormatter.format(y),
    },
    {
        title: 'Validation loss',
        xLabel: 'Step',
        yLabel: 'Loss',
        data,
        color: 'var(--coral)',
        getX: (entry) => entry.step,
        getY: (entry) => entry.val_loss,
        formatY: (y) => lossFormatter.format(y),
    },
    {
        title: 'Validation action error',
        xLabel: 'Step',
        yLabel: 'Mean absolute error',
        data,
        getX: (entry) => entry.step,
        getY: (entry) => entry.val_action_error,
        lines: actionErrorLines(data),
        formatY: (y) => lossFormatter.format(y),
    },
    {
        title: 'Learning rate',
        xLabel: 'Step',
        yLabel: 'Learning rate',
        data,
        getX: (entry) => entry.step,
        getY: (entry) => entry['lr-AdamW'],
        formatY: (y) => learningRateFormatter.format(y),
    },
];

export const JobMetricsContent = ({ jobId }: { jobId: string }) => {
    const query = useJobMetrics(jobId);
    const metricsData = useMemo(() => {
        return query.data ?? [];
    }, [query.data]);

    const metrics = buildMetricSeries(metricsData);

    return (
        <>
            {query.issues.map((issue) => (
                <StatusLight key={issue.id} variant='yellow'>
                    {issue.detail}
                </StatusLight>
            ))}
            <MetricsView series={metrics} isLoading={query.isLoading && query.issues.length === 0} />
        </>
    );
};

const useModelMetrics = (modelId: string) => {
    return useQuery({
        queryKey: getQueryKey(['get', '/api/models/{model_id}/metrics', { params: { path: { model_id: modelId } } }]),
        queryFn: streamedQuery({
            streamFn: (context) => {
                const url = fetchClient.PATH('/api/models/{model_id}/metrics', {
                    params: { path: { model_id: modelId } },
                });

                return batchAsyncIterable(
                    fetchSSE<MetricsEntry>(url, { signal: context.signal }),
                    METRICS_BATCH_INTERVAL_MS
                );
            },
            reducer: foldMetricsBatch,
            initialValue: new Map<number, MetricsEntry>(),
        }),
        select: sortMetricsByStep,
        staleTime: Infinity,
    });
};

export const MetricsContent = ({ modelId }: { modelId: string }) => {
    const query = useModelMetrics(modelId);

    const metricsData = useMemo(() => {
        return query.data ?? [];
    }, [query.data]);

    const metrics = buildMetricSeries(metricsData);

    return <MetricsView series={metrics} isLoading={query.isLoading} />;
};
