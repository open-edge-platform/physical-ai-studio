import { useMemo } from 'react';

import { Grid, Heading, IllustratedMessage } from '@geti-ui/ui';
import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { fetchSSE } from '../../../api/fetch-sse';
import { ReactComponent as EmptyIllustration } from './../../../assets/illustration.svg';
import { MetricGraph } from './metric-graph';

export const NoMetricsAvailable = () => {
    return (
        <IllustratedMessage marginY='size-400'>
            <EmptyIllustration height='250px' />
            <Heading>No metrics available yet</Heading>
        </IllustratedMessage>
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

const filterLossStepMetrics = (data?: MetricsEntry[]) => {
    if (!data) return [];
    return data.flatMap((entry) => {
        // Prefer the per-step train/loss. Fall back to train/loss_step, which ACT
        // logged per-step historically, so jobs still streaming from older runs
        // keep charting.
        const y = entry.train_loss ?? entry.train_loss_step;
        return y == null ? [] : [{ x: entry.step, y }];
    });
};

const filterValidationLossMetrics = (data?: MetricsEntry[]) => {
    if (data == null) return [];

    return data.flatMap((entry) => {
        const y = entry.val_loss;
        return y == null ? [] : [{ x: entry.step, y }];
    });
};

const filterLearningRateMetrics = (data?: MetricsEntry[]) => {
    if (data == null) return [];

    return data.flatMap((entry) => {
        const y = entry['lr-AdamW'];
        return y == null ? [] : [{ x: entry.step, y }];
    });
};

export const JobMetricsContent = ({ jobId }: { jobId: string }) => {
    const query = useQuery({
        queryKey: ['get', '/api/models/{job_id}/model_metrics', jobId],
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

    const lossStepMetrics = useMemo(() => {
        return filterLossStepMetrics(query.data);
    }, [query.data]);

    const validationLossStepMetrics = useMemo(() => {
        return filterValidationLossMetrics(query.data);
    }, [query.data]);

    const learningRateStepMetrics = useMemo(() => {
        return filterLearningRateMetrics(query.data);
    }, [query.data]);

    if (
        [learningRateStepMetrics, validationLossStepMetrics, lossStepMetrics].every((metrics) => metrics.length === 0)
    ) {
        return <NoMetricsAvailable />;
    }

    const metrics = [
        { title: 'Training loss', xLabel: 'Step', yLabel: 'Loss', data: lossStepMetrics },
        { title: 'Validation loss', xLabel: 'Step', yLabel: 'Loss', data: validationLossStepMetrics },
        { title: 'Learning rate', xLabel: 'Step', yLabel: 'Learning rate', data: learningRateStepMetrics },
    ];

    const metricsReadyToRender = metrics.filter((metric) => metric.data.length > 0);

    return (
        <Grid
            columns='repeat(auto-fit, minmax(min(100%, var(--spectrum-global-dimension-size-6000)), 1fr))'
            gap='size-200'
        >
            {metricsReadyToRender.map(({ title, xLabel, yLabel, data }) => (
                <MetricGraph key={title} title={title} yAxisLabel={yLabel} xAxisLabel={xLabel} data={data} />
            ))}
        </Grid>
    );
};

export const MetricsContent = ({ modelId }: { modelId: string }) => {
    const query = useQuery({
        queryKey: ['get', '/api/models/{model_id}/metrics', modelId],
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

    const lossStepMetrics = useMemo(() => {
        return filterLossStepMetrics(query.data);
    }, [query.data]);

    const validationLossStepMetrics = useMemo(() => {
        return filterValidationLossMetrics(query.data);
    }, [query.data]);

    const learningRateStepMetrics = useMemo(() => {
        return filterLearningRateMetrics(query.data);
    }, [query.data]);

    if (
        [learningRateStepMetrics, validationLossStepMetrics, lossStepMetrics].every((metrics) => metrics.length === 0)
    ) {
        return <NoMetricsAvailable />;
    }

    const metrics = [
        { title: 'Training loss', xLabel: 'Step', yLabel: 'Loss', data: lossStepMetrics },
        { title: 'Validation loss', xLabel: 'Step', yLabel: 'Loss', data: validationLossStepMetrics },
        { title: 'Learning rate', xLabel: 'Step', yLabel: 'Learning rate', data: learningRateStepMetrics },
    ];

    const metricsReadyToRender = metrics.filter((metric) => metric.data.length > 0);

    return (
        <Grid
            columns='repeat(auto-fit, minmax(min(100%, var(--spectrum-global-dimension-size-6000)), 1fr))'
            gap='size-200'
        >
            {metricsReadyToRender.map(({ title, xLabel, yLabel, data }) => (
                <MetricGraph key={title} title={title} yAxisLabel={yLabel} xAxisLabel={xLabel} data={data} />
            ))}
        </Grid>
    );
};
