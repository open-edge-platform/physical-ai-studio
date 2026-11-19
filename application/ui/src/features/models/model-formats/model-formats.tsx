import { Suspense, useState } from 'react';

import { Button, DialogContainer, Flex, Grid, Loading, Text } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';
import { Button as AriaButton } from 'react-aria-components';

import { $api } from '../../../api/client';
import type { SchemaModel, SchemaModelExportAction, SchemaModelExportJob } from '../../../api/openapi-spec';
import { JOBS_QUERY_KEY, useJobCache } from '../../jobs/use-job-cache';
import { INFERENCE_BACKENDS, isExportBackend } from '../inference-backends';
import { AddVariantDialog } from './add-variant-dialog';
import { BackendCard } from './backend-card';
import { ExportJobControls } from './export-job-controls';

import classes from './model-formats.module.css';

interface ModelExportsProps {
    model: SchemaModel;
}

const cardGridColumns = 'repeat(auto-fill, minmax(min(100%, var(--spectrum-global-dimension-size-4600)), 1fr))';
const backendOrder = ['torch', 'openvino', 'onnx', 'executorch'];

const ModelFormatsContents = ({ model }: { model: SchemaModel }) => {
    const [creationBackends, setCreationBackends] = useState<SchemaModelExportAction['backend'][] | null>(null);
    const { data: modelDetail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });
    const jobs = $api.useQuery('get', '/api/jobs');
    const { addJob } = useJobCache();
    const exportModel = $api.useMutation('post', '/api/models/{model_id}/exports/{backend}', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const modelJobs = (jobs.data ?? [])
        .filter((job): job is SchemaModelExportJob => job.type === 'model_export' && job.payload.model_id === model.id)
        .toSorted((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
    const isExporting =
        exportModel.isPending || modelJobs.some(({ status }) => status === 'pending' || status === 'running');
    const requestedBackend = exportModel.variables?.params.path.backend;
    // Job history does not prove an artifact still exists: exports can be removed independently.
    const incompleteJobs = modelJobs.filter(
        (job, index) =>
            job.status !== 'completed' &&
            modelJobs.findIndex(({ payload }) => payload.backend === job.payload.backend) === index
    );
    const backends = [
        ...new Set([
            ...modelDetail.exports.map(({ type }) => type),
            ...incompleteJobs.map(({ payload }) => payload.backend),
            ...(requestedBackend && (exportModel.isPending || exportModel.isError) ? [requestedBackend] : []),
        ]),
    ]
        .filter(isExportBackend)
        .toSorted((a, b) => backendOrder.indexOf(a) - backendOrder.indexOf(b));
    const actions = (modelDetail.available_actions ?? [])
        .filter(({ backend, operation }) => operation === 'export' && !backends.includes(backend))
        .toSorted((a, b) => backendOrder.indexOf(a.backend) - backendOrder.indexOf(b.backend));
    const isExportDisabled = jobs.isPending || jobs.isError || isExporting;
    const startExport = (backend: (typeof backends)[number], onSuccess?: () => void) => {
        exportModel.mutate({ params: { path: { model_id: model.id!, backend } } }, { onSuccess });
    };

    return (
        <Flex direction='column' gap='size-200' marginTop='size-400'>
            {jobs.isError && (
                <Flex gap='size-100' alignItems='center'>
                    <Text>Unable to load export status.</Text>
                    <Button variant='secondary' onPress={() => jobs.refetch()}>
                        Retry job status
                    </Button>
                </Flex>
            )}
            {jobs.isPending && <div role='status'>Loading export status…</div>}
            {backends.length === 0 && (
                <Text>
                    {actions.length > 0
                        ? 'No variants yet. Add a variant to export the stored checkpoint.'
                        : 'No exported variants are available for this model.'}
                </Text>
            )}
            <Grid
                gap='size-200'
                UNSAFE_style={{
                    gridTemplateColumns: cardGridColumns,
                }}
            >
                {backends.map((backendType) => {
                    const isAvailable = modelDetail.exports.some(({ type }) => type === backendType);
                    const job =
                        modelJobs.find(
                            ({ payload, status }) =>
                                payload.backend === backendType && (status === 'pending' || status === 'running')
                        ) ?? modelJobs.find(({ payload }) => payload.backend === backendType);
                    const isSubmitting = exportModel.isPending && requestedBackend === backendType;
                    const error = requestedBackend === backendType ? exportModel.error : undefined;
                    const canRetry = modelDetail.available_actions?.some(
                        ({ backend, operation }) => backend === backendType && operation === 'export'
                    );
                    return (
                        <section
                            key={backendType}
                            aria-label={`${INFERENCE_BACKENDS[backendType].label} variant`}
                            aria-busy={isSubmitting || job?.status === 'pending' || job?.status === 'running'}
                        >
                            <BackendCard backendType={backendType} model={model} modelDetail={modelDetail}>
                                {!isAvailable && (
                                    <ExportJobControls
                                        key={job?.id}
                                        label={INFERENCE_BACKENDS[backendType].label}
                                        job={error ? undefined : job}
                                        isSubmitting={isSubmitting}
                                        error={error}
                                        onRetry={canRetry ? () => startExport(backendType) : undefined}
                                        isRetryDisabled={isExportDisabled}
                                    />
                                )}
                            </BackendCard>
                        </section>
                    );
                })}
                {actions.length > 0 && (
                    <AriaButton
                        aria-label='Add variant'
                        className={classes.addCard}
                        isDisabled={isExportDisabled}
                        onPress={() => setCreationBackends(actions.map(({ backend }) => backend))}
                    >
                        <span className={classes.addTitle}>
                            <Add aria-hidden='true' />
                            <span>Add variant</span>
                        </span>
                        <span>Export to another backend</span>
                    </AriaButton>
                )}
            </Grid>
            <DialogContainer
                isKeyboardDismissDisabled={exportModel.isPending}
                onDismiss={() => {
                    if (!exportModel.isPending) setCreationBackends(null);
                }}
            >
                {creationBackends !== null && (
                    <AddVariantDialog
                        backends={creationBackends}
                        isSubmitting={exportModel.isPending}
                        isExportDisabled={isExportDisabled}
                        error={exportModel.error}
                        onExport={(backend) => startExport(backend, () => setCreationBackends(null))}
                        close={() => setCreationBackends(null)}
                    />
                )}
            </DialogContainer>
        </Flex>
    );
};

export const ModelFormats = ({ model }: ModelExportsProps) => {
    return (
        <Suspense fallback={<Loading mode='inline' size='M' marginY='size-400' />}>
            <ModelFormatsContents model={model} />
        </Suspense>
    );
};
