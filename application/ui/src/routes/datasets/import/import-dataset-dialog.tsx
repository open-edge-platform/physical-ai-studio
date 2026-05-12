import { Suspense, useEffect, useEffectEvent, useState } from 'react';

import { Content, Dialog, Divider, Heading, Loading } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';

import { SchemaDatasetImportJob } from '../../../api/openapi-spec';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { ImportStepDetectionFailed } from './import-step-detection-failed';
import { ImportStepInProgress } from './import-step-in-progress';
import { ImportStepUpload } from './import-step-upload';
import { ImportStepUserReview } from './import-step-user-review';
import { useDatasetImportJobQuery, type FinalizeFields } from './use-dataset-import-job-state';

const DETECTION_STEPS = ['queued_for_detection', 'detecting_format', 'building_manifest_draft'];

const IMPORT_DIALOG_VIEW_STATE = {
    UPLOAD: 'upload',
    DETECTING: 'detecting',
    DETECTION_FAILED: 'detection_failed',
    USER_REVIEW: 'user_review',
    IMPORTING: 'importing',
} as const;

type ImportDialogViewState = (typeof IMPORT_DIALOG_VIEW_STATE)[keyof typeof IMPORT_DIALOG_VIEW_STATE];

interface InternalImportDatasetDialogProps {
    project_id: string;
    onClose: () => void;
    initialJobId?: string;
    onPendingJobDismissed?: (jobId: string) => void;
    onImportCompleted?: (datasetId: string) => void;
}

interface ImportDatasetDialogProps {
    onClose: () => void;
    initialJobId?: string;
    onPendingJobDismissed?: (jobId: string) => void;
    onImportCompleted?: (datasetId: string) => void;
}

const DatasetImportHeading = ({
    importJob,
    viewState,
}: {
    importJob: SchemaDatasetImportJob | undefined;
    viewState: ImportDialogViewState;
}) => {
    if (viewState !== IMPORT_DIALOG_VIEW_STATE.DETECTION_FAILED) {
        return <Heading>Import dataset</Heading>;
    }

    const formatHint = importJob?.payload?.format_hint;
    const usedAutoDetection = formatHint === 'auto' || formatHint === undefined;

    return (
        <Heading>
            {usedAutoDetection ? 'Automatic format detection failed' : 'Selected format validation failed'}
        </Heading>
    );
};

const getDialogViewStatus = (importJob: SchemaDatasetImportJob | undefined): ImportDialogViewState => {
    const importPayload = importJob?.payload;
    const payloadStep = importPayload?.step;

    if (importJob === undefined || payloadStep === 'awaiting_archive_upload') {
        return IMPORT_DIALOG_VIEW_STATE.UPLOAD;
    }

    const isDetectionStep = payloadStep !== undefined && DETECTION_STEPS.includes(String(payloadStep));
    if (importJob.status === 'failed' && isDetectionStep) {
        return IMPORT_DIALOG_VIEW_STATE.DETECTION_FAILED;
    }

    if (isDetectionStep) {
        return IMPORT_DIALOG_VIEW_STATE.DETECTING;
    }

    if (payloadStep === 'awaiting_user_review') {
        return IMPORT_DIALOG_VIEW_STATE.USER_REVIEW;
    }

    if (payloadStep === 'queued_for_import' || payloadStep === 'importing_dataset' || payloadStep === 'completed') {
        return IMPORT_DIALOG_VIEW_STATE.IMPORTING;
    }

    return IMPORT_DIALOG_VIEW_STATE.UPLOAD;
};

const useOnImportDone = (importJob: SchemaDatasetImportJob | undefined, onImportDone: (datasetId: string) => void) => {
    const completedDatasetId = importJob?.status === 'completed' ? importJob.payload?.result_dataset_id : undefined;
    const onImportDoneEvent = useEffectEvent(onImportDone);

    useEffect(() => {
        if (completedDatasetId) {
            onImportDoneEvent(completedDatasetId);
        }
    }, [completedDatasetId, onImportDoneEvent]);
};

const InternalImportDatasetDialog = ({
    project_id,
    onClose,
    initialJobId,
    onPendingJobDismissed,
    onImportCompleted,
}: InternalImportDatasetDialogProps) => {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [datasetName, setDatasetName] = useState('');
    const [finalizeFields, setFinalizeFields] = useState<FinalizeFields>({
        defaultTask: '',
        environmentId: undefined,
    });
    const [importJobId, setImportJobId] = useState<string | undefined>(initialJobId);

    const importJobQuery = useDatasetImportJobQuery(importJobId);
    const importJob = importJobQuery.data as SchemaDatasetImportJob | undefined;

    const status = getDialogViewStatus(importJob);

    const onDialogClose = () => {
        if (importJobId && status === IMPORT_DIALOG_VIEW_STATE.USER_REVIEW) {
            onPendingJobDismissed?.(importJobId);
        }
        onClose();
    };

    useOnImportDone(importJob, (datasetId: string) => {
        void queryClient.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}'] });
        void queryClient.invalidateQueries({ queryKey: ['get', '/api/jobs'] });
        onImportCompleted?.(datasetId);
        onDialogClose();
        navigate(paths.project.datasets.show({ project_id, dataset_id: datasetId }));
    });

    return (
        <Dialog>
            <DatasetImportHeading importJob={importJob} viewState={status} />
            <Divider />

            {status === IMPORT_DIALOG_VIEW_STATE.UPLOAD && (
                <ImportStepUpload
                    importJob={importJob}
                    project_id={project_id}
                    onClose={onDialogClose}
                    datasetName={datasetName}
                    onDatasetNameChange={setDatasetName}
                    onUploaded={setImportJobId}
                />
            )}

            {importJob !== undefined && status === IMPORT_DIALOG_VIEW_STATE.DETECTING && (
                <ImportStepInProgress statusMessage='Upload accepted. Waiting for server-side dataset detection...' />
            )}

            {importJob !== undefined && status === IMPORT_DIALOG_VIEW_STATE.IMPORTING && (
                <ImportStepInProgress statusMessage={importJob?.message ?? 'Importing dataset...'} />
            )}

            {importJob !== undefined && status === IMPORT_DIALOG_VIEW_STATE.DETECTION_FAILED && (
                <ImportStepDetectionFailed importJob={importJob} onClose={onDialogClose} />
            )}

            {importJob !== undefined && status === IMPORT_DIALOG_VIEW_STATE.USER_REVIEW && importJobId && (
                <ImportStepUserReview
                    importJob={importJob}
                    project_id={project_id}
                    onClose={onClose}
                    fields={finalizeFields}
                    onFieldsChange={setFinalizeFields}
                />
            )}
        </Dialog>
    );
};

export const ImportDatasetDialog = ({
    onClose,
    initialJobId,
    onPendingJobDismissed,
    onImportCompleted,
}: ImportDatasetDialogProps) => {
    const { project_id } = useProjectId();

    return (
        <Suspense
            fallback={
                <Dialog>
                    <Heading>Import dataset</Heading>
                    <Divider />
                    <Content>
                        <Loading />
                    </Content>
                </Dialog>
            }
        >
            <InternalImportDatasetDialog
                project_id={project_id}
                onClose={onClose}
                initialJobId={initialJobId}
                onPendingJobDismissed={onPendingJobDismissed}
                onImportCompleted={onImportCompleted}
            />
        </Suspense>
    );
};

export type { ImportDatasetDialogProps };
