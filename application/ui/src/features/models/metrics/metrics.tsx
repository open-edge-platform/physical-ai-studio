import { useId, useMemo, useState } from 'react';

import { Grid, Heading, IllustratedMessage, Loading, StatusLight } from '@geti-ui/ui';
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
import { hasMetricValue, type MetricLine, type MetricsEntry } from './types';

const NoMetricsAvailable = () => {
    return (
        <IllustratedMessage marginY='size-400'>
            <EmptyIllustration height='250px' />
            <Heading>No metrics available yet</Heading>
        </IllustratedMessage>
    );
};

export interface MetricSeries {
    title: string;
    helpText?: string;
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

    const seriesReadyToRender = series.filter((metric) =>
        metric.data.some((entry) => hasMetricValue(entry, metric.getY, metric.lines))
    );

    if (seriesReadyToRender.length === 0) {
        return <NoMetricsAvailable />;
    }

    return (
        <Grid
            columns='repeat(auto-fit, minmax(min(100%, var(--spectrum-global-dimension-size-6000)), 1fr))'
            gap='size-200'
        >
            {seriesReadyToRender.map(({ title, helpText, xLabel, yLabel, data, color, getX, getY, lines, formatY }) => (
                <MetricGraph
                    key={title}
                    syncId={syncId}
                    title={title}
                    helpText={helpText}
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
const SERIES_COLORS = [
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
            primary: true,
            getY: (entry) => entry.val_action_error,
        },
        ...[...keys].map((key, index) => ({
            name: key.slice(ACTION_ERROR_PREFIX.length).replace(/\.pos$/, ''),
            color: SERIES_COLORS[index % SERIES_COLORS.length],
            getY: (entry: MetricsEntry) => entry[key],
        })),
    ];
};

const learningRateLines = (data: MetricsEntry[]): MetricLine[] => {
    const keys = new Set(
        data.flatMap((entry) => Object.keys(entry).filter((key) => key.startsWith('lr-') && entry[key] != null))
    );

    return [...keys].sort().map((key, index) => ({
        name: key.slice('lr-'.length),
        color: index === 0 ? 'var(--energy-blue)' : SERIES_COLORS[(index - 1) % SERIES_COLORS.length],
        getY: (entry) => entry[key],
    }));
};

export const buildMetricSeries = (data: MetricsEntry[]): MetricSeries[] => {
    const learningRates = learningRateLines(data);

    return [
        {
            title: 'Training loss',
            helpText:
                "The policy's optimization loss on training demonstrations. Use the trend to track fitting; " +
                'the scale depends on the policy, so values are not directly comparable across policy types.',
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
            helpText:
                'Loss on held-out demonstrations, not task success. It can rise while the policy improves, ' +
                'especially for flow-matching policies such as SmolVLA and Pi0.5. ' +
                'Evaluate task success on your robot or in simulation.',
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
            helpText:
                'Mean absolute difference between predicted and recorded actions, in dataset units. ' +
                'The dashed Mean line averages across joints. For SO-101, joint positions range from -100 to 100 ' +
                'and the gripper from 0 to 100. Lower error does not guarantee task success.',
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
            helpText:
                'Learning rate used by the optimizer at each recorded training step. ' +
                'Separate lines represent optimizers or parameter groups, which can use different rates and schedules.',
            xLabel: 'Step',
            yLabel: 'Learning rate',
            data,
            getX: (entry) => entry.step,
            getY: (entry) => learningRates[0]?.getY(entry),
            lines: learningRates,
            formatY: (y) => learningRateFormatter.format(y),
        },
    ];
};

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
