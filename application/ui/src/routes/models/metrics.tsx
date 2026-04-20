import { experimental_streamedQuery as streamedQuery, queryOptions, useQuery } from '@tanstack/react-query';
import { fetchClient } from '../../api/client';
import { fetchSSE } from '../../api/fetch-sse';
import { useMemo } from 'react';
import { MetricGraph } from './metrics-graph.component';

interface MetricsEntry {
    epoch: number;
    step: number;
    train_loss: number | null;
    train_loss_step: number | null;
}

const filterLossStepMetrics = (data?: MetricsEntry[]) => {
    if (!data) return [];
    const stepRows = data.filter((entry): entry is MetricsEntry => {
        return (
            entry.train_loss !== null
        )
    });
    return stepRows.map((row) => ({x: row.step, y: row.train_loss!}))
}

export const JobMetricsContent = ({jobId}: {jobId: string}) => {
    const query = useQuery(
        queryOptions({
            queryKey: ['get', '/api/models/{job_id}/metrics', jobId],
            queryFn: streamedQuery({
                queryFn: (context) => {
                    const url = fetchClient.PATH('/api/jobs/{job_id}/model_metrics', {
                        params: { path: { job_id: jobId } },
                    });

                    return fetchSSE<MetricsEntry>(url, { signal: context.signal });
                },
            }),
            staleTime: Infinity,
        })
    );

    const lossStepMetrics = useMemo(() => {
        return filterLossStepMetrics(query.data);
    }, [query.data])

    return (
        <MetricGraph title={"Loss"} yAxisLabel={"Loss"} xAxisLabel='Step' data={lossStepMetrics}/>
    )
}

export const MetricsContent = ({modelId}: {modelId: string}) => {

    const query = useQuery(
        queryOptions({
            queryKey: ['get', '/api/models/{model_id}/metrics', modelId],
            queryFn: streamedQuery({
                queryFn: (context) => {
                    const url = fetchClient.PATH('/api/models/{model_id}/metrics', {
                        params: { path: { model_id: modelId } },
                    });

                    return fetchSSE<MetricsEntry>(url, { signal: context.signal });
                },
            }),
            staleTime: Infinity,
        })
    );

    const lossStepMetrics = useMemo(() => {
        return filterLossStepMetrics(query.data);
    }, [query.data])

    return (
        <MetricGraph title={"Loss"} yAxisLabel={"Loss"} xAxisLabel='Step' data={lossStepMetrics}/>
    )
}
