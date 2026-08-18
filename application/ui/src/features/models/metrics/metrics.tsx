import { useMemo } from 'react';

import { experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { fetchSSE } from '../../../api/fetch-sse';
import { MetricGraph } from './metric-graph';

interface MetricsEntry {
    epoch: number;
    step: number;
    train_loss: number | null | undefined;
    train_loss_step: number | null | undefined;
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

    return <MetricGraph title={'Loss'} yAxisLabel={'Loss'} xAxisLabel='Step' data={lossStepMetrics} />;
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

    return <MetricGraph title={'Loss'} yAxisLabel={'Loss'} xAxisLabel='Step' data={lossStepMetrics} />;
};
