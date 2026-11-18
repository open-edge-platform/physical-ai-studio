import { Suspense, useState } from 'react';

import { Button, DialogContainer, Flex, Grid, Loading, Text } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';
import { Button as AriaButton } from 'react-aria-components';

import { $api } from '../../../api/client';
import type {
    SchemaExportVariantDetail,
    SchemaModel,
    SchemaModelExportAction,
    SchemaModelExportJob,
} from '../../../api/openapi-spec';
import { JOBS_QUERY_KEY, useJobCache } from '../../jobs/use-job-cache';
import { INFERENCE_BACKENDS, isExportBackend } from '../inference-backends';
import { AddModelFormatDialog } from './add-model-format-dialog';
import { BackendCard } from './backend-card';
import { ExportJobControls } from './export-job-controls';

import classes from './model-formats.module.css';

interface ModelExportsProps {
    model: SchemaModel;
}

const cardGridColumns = 'repeat(auto-fill, minmax(min(100%, var(--spectrum-global-dimension-size-4600)), 1fr))';
const backendOrder = ['torch', 'openvino', 'onnx', 'executorch'];

interface VariantTarget {
    backend: SchemaModelExportAction['backend'];
    variant: SchemaExportVariantDetail['id'];
}

const actionTarget = (action: SchemaModelExportAction): VariantTarget => ({
    backend: action.backend,
    variant: action.operation === 'int8_sym' ? 'int8' : 'original',
});

const targetKey = ({ backend, variant }: VariantTarget) => `${backend}/${variant}`;

const ModelFormatsContents = ({ model }: { model: SchemaModel }) => {
    const [creation, setCreation] = useState<{
        actions: SchemaModelExportAction[];
        initialAction?: SchemaModelExportAction;
    } | null>(null);
    const [requestedAction, setRequestedAction] = useState<SchemaModelExportAction | null>(null);
    const { data: modelDetail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });
    const jobs = $api.useQuery('get', '/api/jobs');
    const { addJob } = useJobCache();
    const exportModel = $api.useMutation('post', '/api/models/{model_id}/exports/{backend}', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const optimizeModel = $api.useMutation('post', '/api/models/{model_id}/optimize', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const submission = requestedAction?.operation === 'int8_sym' ? optimizeModel : exportModel;
    const modelJobs = (jobs.data ?? [])
        .filter((job): job is SchemaModelExportJob => job.type === 'model_export' && job.payload.model_id === model.id)
        .toSorted((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
    const isExporting =
        submission.isPending || modelJobs.some(({ status }) => status === 'pending' || status === 'running');
    // Job history does not prove an artifact still exists: exports can be removed independently.
    const incompleteJobs = modelJobs.filter(
        (job, index) =>
            job.status !== 'completed' &&
            job.status !== 'canceled' &&
            modelJobs.findIndex(
                ({ payload }) => targetKey(actionTarget(payload)) === targetKey(actionTarget(job.payload))
            ) === index
    );
    const candidates: VariantTarget[] = [
        ...modelDetail.exports.flatMap(({ type, variants }) =>
            isExportBackend(type) ? (variants ?? []).map(({ id }) => ({ backend: type, variant: id })) : []
        ),
        ...incompleteJobs.map(({ payload }) => actionTarget(payload)),
        ...(requestedAction && (submission.isPending || submission.isError) ? [actionTarget(requestedAction)] : []),
    ];
    const cards = candidates
        .filter(
            (target, index) => candidates.findIndex((candidate) => targetKey(candidate) === targetKey(target)) === index
        )
        .toSorted(
            (a, b) =>
                backendOrder.indexOf(a.backend) - backendOrder.indexOf(b.backend) ||
                Number(a.variant !== 'original') - Number(b.variant !== 'original')
        );
    const actions = (modelDetail.available_actions ?? [])
        .filter((action) => !cards.some((target) => targetKey(target) === targetKey(actionTarget(action))))
        .toSorted((a, b) => backendOrder.indexOf(a.backend) - backendOrder.indexOf(b.backend));
    const isExportDisabled = jobs.isPending || jobs.isError || isExporting;
    const startExport = (action: SchemaModelExportAction, onSuccess?: () => void) => {
        setRequestedAction(action);
        if (action.operation === 'int8_sym') {
            optimizeModel.mutate({ params: { path: { model_id: model.id! } } }, { onSuccess });
        } else {
            exportModel.mutate({ params: { path: { model_id: model.id!, backend: action.backend } } }, { onSuccess });
        }
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
            {cards.length === 0 && (
                <Text>
                    {actions.length > 0
                        ? 'No model formats yet. Add a model format to export the stored checkpoint.'
                        : 'No exported model formats are available for this model.'}
                </Text>
            )}
            <Grid
                gap='size-200'
                UNSAFE_style={{
                    gridTemplateColumns: cardGridColumns,
                }}
            >
                {cards.map((target) => {
                    const backendType = target.backend;
                    const variant = modelDetail.exports
                        .find(({ type }) => type === backendType)
                        ?.variants?.find(({ id }) => id === target.variant);
                    const matchingJobs = modelJobs.filter(
                        ({ payload }) => targetKey(actionTarget(payload)) === targetKey(target)
                    );
                    const job =
                        matchingJobs.find(({ status }) => status === 'pending' || status === 'running') ??
                        matchingJobs[0];
                    const isRequested =
                        requestedAction !== null && targetKey(actionTarget(requestedAction)) === targetKey(target);
                    const isSubmitting = submission.isPending && isRequested;
                    const isExportInProgress =
                        !variant && (isSubmitting || job?.status === 'pending' || job?.status === 'running');
                    const error = isRequested ? submission.error : undefined;
                    const retryAction = modelDetail.available_actions?.find(
                        (action) => targetKey(actionTarget(action)) === targetKey(target)
                    );
                    const optimizeAction =
                        target.variant === 'original'
                            ? actions.find(
                                  (action) => action.backend === backendType && action.operation === 'int8_sym'
                              )
                            : undefined;
                    const statusLabel =
                        INFERENCE_BACKENDS[backendType].label + (target.variant === 'int8' ? ' INT8' : '');
                    return (
                        <BackendCard
                            key={targetKey(target)}
                            backendType={backendType}
                            model={model}
                            variant={variant}
                            variantId={target.variant}
                            isExporting={isExportInProgress}
                            action={optimizeAction}
                            isCreateDisabled={isExportDisabled}
                            onCreate={(initialAction) => setCreation({ actions, initialAction })}
                        >
                            {!variant && (
                                <ExportJobControls
                                    key={job?.id}
                                    label={statusLabel}
                                    job={error ? undefined : job}
                                    isSubmitting={isSubmitting}
                                    error={error}
                                    onRetry={retryAction ? () => startExport(retryAction) : undefined}
                                    isRetryDisabled={isExportDisabled}
                                />
                            )}
                        </BackendCard>
                    );
                })}
                {actions.length > 0 && (
                    <AriaButton
                        aria-label='Add model format'
                        className={classes.addCard}
                        isDisabled={isExportDisabled}
                        onPress={() => setCreation({ actions })}
                    >
                        <span className={classes.addTitle}>
                            <Add aria-hidden='true' />
                            <span>Add model format</span>
                        </span>
                        <span>Export to another inference backend</span>
                    </AriaButton>
                )}
            </Grid>
            <DialogContainer
                isKeyboardDismissDisabled={submission.isPending}
                onDismiss={() => {
                    if (!submission.isPending) setCreation(null);
                }}
            >
                {creation !== null && (
                    <AddModelFormatDialog
                        actions={creation.actions}
                        initialAction={creation.initialAction}
                        isSubmitting={submission.isPending}
                        isExportDisabled={isExportDisabled}
                        error={submission.error}
                        onExport={(action) => startExport(action, () => setCreation(null))}
                        close={() => setCreation(null)}
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
