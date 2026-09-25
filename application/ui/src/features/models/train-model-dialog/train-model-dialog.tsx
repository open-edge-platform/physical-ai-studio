import { useEffect, useMemo, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Key, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaTrainJob as SchemaJob, SchemaModel } from '../../../api/openapi-spec';
import { useProject } from '../../projects/use-project';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { getDisplayHealth, healthLabel, healthVariant } from '../../training-targets/remote-trainer-health-utils';
import { useRemoteTrainersHealth } from '../../training-targets/training-targets-table/use-remote-trainers-health';
import { useRemoteTrainerHealth } from '../../training-targets/use-remote-trainer-health';
import { supportsLora } from '../shared/peft';
import { supportsSnapflow } from '../shared/snapflow';
import { ExportStep } from './export-step';
import { FeatureMappingStep } from './feature-mapping-step';
import { formatBytes, MODELS } from './policies';
import { SetupStep } from './setup-step';
import { TrainingDeviceInfo } from './training-device-info';
import { MIN_EPOCHS_FOR_SNAPFLOW, TrainingParameters } from './training-parameters';
import { TrainingSummaryNote } from './training-summary-note';
import { useExportBackends } from './use-export-backends';
import { useFeatureMapping } from './use-feature-mapping';
import { pickBestDevice, useBestTrainingDevice } from './use-training-devices';
import { getWizardSteps, WizardStep } from './wizard-steps';

export type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaJob['payload'];
};

interface TrainModelDialogProps {
    baseModel?: SchemaModel;
    close: (job: SchemaJob | undefined) => void;
    defaultMaxEpochs?: number;
}

export type TrainingTargetKind = 'local' | 'trainer';

export type TrainingTargetStatusVariant = 'positive' | 'notice' | 'negative' | 'neutral' | 'yellow';

export type TrainingTargetOption = {
    id: string;
    label: string;
    /** `local` or `trainer:<remote_trainer_id>`. */
    kind: TrainingTargetKind;
    statusVariant: TrainingTargetStatusVariant;
    statusLabel: string;
};

const LOCAL_TARGET_ID = 'local';

/** Mirrors `_DEFAULT_SNAPFLOW_DISTILL_EPOCHS` in the backend payload schema. */
const DEFAULT_SNAPFLOW_DISTILL_EPOCHS = 3;

/** Strip the `trainer:` prefix off a training-target option id. */
const targetRawId = (id: string): string => id.split(':', 2)[1] ?? id;

export const TrainModelDialog = ({ baseModel, close, defaultMaxEpochs = 5 }: TrainModelDialogProps) => {
    const bestDevice = useBestTrainingDevice();
    const { data: remoteTrainers = [] } = $api.useQuery('get', '/api/remote-trainers');
    // Continuing an existing model needs its checkpoint, which only this machine
    // has: the trainer protocol can receive a dataset but not a base checkpoint.
    // So a resumed run offers local training only.
    const canTrainRemotely = baseModel === undefined;
    const remoteTrainerHealthById = useRemoteTrainersHealth(canTrainRemotely ? remoteTrainers.map((t) => t.id) : []);
    // One control lists every target type (local, direct-URL trainer - which
    // also covers an SSH-tunneled trainer) rather than a local/remote mode
    // toggle, so the derived `training_target` is always unambiguous. Each
    // option carries its own status variant/label so the "Run on" dropdown
    // shows, at a glance, which targets are currently working correctly.
    const trainingTargetOptions: TrainingTargetOption[] = [
        {
            id: LOCAL_TARGET_ID,
            label: 'This machine (local)',
            kind: 'local',
            statusVariant: 'positive',
            statusLabel: bestDevice ? bestDevice.type.toUpperCase() : 'CPU only',
        },
        ...(canTrainRemotely
            ? remoteTrainers.map((remoteTrainer) => {
                  const entry = remoteTrainerHealthById.get(remoteTrainer.id);
                  const displayHealth = getDisplayHealth(remoteTrainer.id, entry?.health, entry?.hasError ?? false);
                  const isChecking = entry?.isChecking ?? false;
                  return {
                      id: `trainer:${remoteTrainer.id}`,
                      label: remoteTrainer.name,
                      kind: 'trainer' as const,
                      statusVariant: healthVariant(displayHealth, isChecking),
                      statusLabel: healthLabel(displayHealth, isChecking),
                  };
              })
            : []),
    ];

    const defaultDatasetId = baseModel?.dataset_id ?? null;
    const extraPayload = baseModel ? { base_model_id: baseModel.id! } : undefined;

    const [currentStep, setCurrentStep] = useState<WizardStep>('setup');
    const [selectedPolicy, setSelectedPolicy] = useState<string>(baseModel?.policy ?? 'act');
    const { datasets, id: projectId } = useProject();

    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxEpochs, setMaxEpochs] = useState<number>(defaultMaxEpochs);
    const [batchSize, setBatchSize] = useState<number>(8);
    const [numWorkers, setNumWorkers] = useState<Key | null>('auto');
    const [autoScaleBatchSize, setAutoScaleBatchSize] = useState<boolean>(false);
    const [precision, setPrecision] = useState<Key | null>(bestDevice?.type === 'cuda' ? 'bf16-mixed' : '32-true');
    const [compileModel, setCompileModel] = useState<boolean>(false);
    const [loraEnabled, setLoraEnabled] = useState<boolean>(false);
    const [loraRank, setLoraRank] = useState<number>(32);
    const [loraAlpha, setLoraAlpha] = useState<number | null>(null);
    const [loraDropout, setLoraDropout] = useState<number>(0.05);
    const [loraUseDora, setLoraUseDora] = useState<boolean>(false);
    const [snapflowEnabled, setSnapflowEnabled] = useState<boolean>(false);
    const [snapflowDistillEpochs, setSnapflowDistillEpochs] = useState<number>(DEFAULT_SNAPFLOW_DISTILL_EPOCHS);
    const [augmentImages, setAugmentImages] = useState<boolean>(false);
    const [targetId, setTargetId] = useState<Key | null>(LOCAL_TARGET_ID);
    const selectedTarget = trainingTargetOptions.find((option) => option.id === targetId) ?? null;
    const isRemoteTarget = selectedTarget?.kind === 'trainer';
    const isLoraSupported = supportsLora(selectedPolicy);
    const isLoraRequested = isLoraSupported && loraEnabled;
    const isSnapflowSupported = supportsSnapflow(selectedPolicy);
    // snapflow_distill_epochs is additive on top of max_epochs (the teacher phase
    // always runs the full max_epochs before distillation extends the run), so it
    // needs no clamp against max_epochs.
    const isSnapflowRequested = isSnapflowSupported && snapflowEnabled && maxEpochs >= MIN_EPOCHS_FOR_SNAPFLOW;
    const {
        health: remoteTrainerHealth,
        isChecking: isCheckingRemoteTrainer,
        checkHealth: checkRemoteTrainerHealth,
    } = useRemoteTrainerHealth(isRemoteTarget ? targetRawId(selectedTarget.id) : null);
    const remoteUnavailable =
        isRemoteTarget &&
        (remoteTrainerHealth?.status === 'unreachable' ||
            remoteTrainerHealth?.status === 'starting' ||
            remoteTrainerHealth?.reason_code === 'docker_unavailable' ||
            remoteTrainerHealth?.reason_code === 'accelerator_unavailable' ||
            remoteTrainerHealth?.reason_code === 'container_accelerator_unavailable');
    const {
        data: policyAccess,
        isLoading: isCheckingPolicyAccess,
        isError: policyAccessCheckFailed,
    } = $api.useQuery('get', '/api/policies/{policy}/huggingface-access', {
        params: { path: { policy: selectedPolicy } },
    });
    // Fail closed: a policy with a *required* Hub dependency (see
    // `_HUGGINGFACE_REQUIREMENTS`) that we could not verify access for -
    // because the check is still loading, or because the request itself
    // failed outright (as opposed to the backend reporting a definite
    // `unavailable` status for one repository) - must not silently let
    // training through only to fail deep into a remote run instead.
    const hasRequiredHuggingFaceDependency =
        policyAccess?.requirements.some((requirement) => requirement.required) ?? true;
    const policyAccessBlocksTraining =
        isCheckingPolicyAccess ||
        (policyAccessCheckFailed && hasRequiredHuggingFaceDependency) ||
        policyAccess?.requirements.some(
            (requirement) =>
                requirement.required && (requirement.status === 'missing_token' || requirement.status === 'denied')
        ) === true;
    const bestRemoteDevice = useMemo(() => pickBestDevice(remoteTrainerHealth?.devices ?? []), [remoteTrainerHealth]);
    // The device actually driving this job: the local GPU when training locally,
    // or the remote trainer's reported GPU once its health check resolves. Auto
    // scale/precision defaults and the disabled state below should track
    // whichever one is currently in play.
    const activeDevice = useMemo(() => {
        if (isRemoteTarget) {
            return bestRemoteDevice;
        }
        return bestDevice;
    }, [isRemoteTarget, bestRemoteDevice, bestDevice]);

    useEffect(() => {
        if (activeDevice?.type === 'cuda') {
            setPrecision('bf16-mixed');
        } else {
            setPrecision('32-true');
        }
    }, [activeDevice]);

    // Cameras are mapped onto the policy's fixed camera slots; the mapping lives
    // here so the step can be left and re-entered without losing it.
    const featureMapping = useFeatureMapping(selectedPolicy, selectedDataset?.toString());

    // Which formats the trained model is exported to; the policy decides what is
    // on offer.
    const exportSelection = useExportBackends(selectedPolicy);

    const trainMutation = $api.useMutation('post', '/api/jobs:train', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
    });
    // `save` awaits health re-checks before it ever calls `mutateAsync`, so
    // `trainMutation.isPending` alone doesn't cover the whole submission window
    // — a double-click (or the dialog's slow close after success) can start a
    // second `save()` call while the first one is still awaiting those checks.
    // Track submission with its own flag so a second call is a no-op for the
    // entire duration, not just while the mutation itself is in flight.
    const [isSubmitting, setIsSubmitting] = useState(false);
    // Surfaced when the final remote-trainer health check or job submission fails.
    const [submitError, setSubmitError] = useState<string | null>(null);

    // Everything the job needs is picked on the setup step, so that is the only
    // step that can block progress; the later steps are free to be skipped through.
    const isSetupIncomplete =
        !selectedDataset || !selectedPolicy || targetId === null || remoteUnavailable || policyAccessBlocksTraining;

    // A policy without a fixed camera order has no feature-mapping step at all,
    // and neither does retraining, so the steps -- and their numbering -- follow
    // the selected policy.
    const isRetraining = baseModel !== undefined;
    const steps = useMemo(() => getWizardSteps(selectedPolicy, isRetraining), [selectedPolicy, isRetraining]);
    const hasFeatureMapping = steps.includes('feature-mapping');
    // Switching to a policy that skips a step can leave `currentStep` behind, and
    // the setup step is the only one every policy has.
    const activeStep = steps.includes(currentStep) ? currentStep : 'setup';
    const currentStepIndex = steps.indexOf(activeStep);
    const isLastStep = currentStepIndex === steps.length - 1;

    const isStepBlocked =
        isSetupIncomplete ||
        // Checked on every step from the mapping on, not just its own: the dataset's
        // cameras can finish loading after Next was pressed and turn the mapping invalid.
        (hasFeatureMapping && activeStep !== 'setup' && featureMapping.error !== null) ||
        (activeStep === 'export' && exportSelection.error !== null);

    // What the later steps recap: the trainer that runs the job and the device it
    // trains on, plus the dataset and policy the run is about.
    const trainingTargetLabel = selectedTarget?.label ?? 'This machine (local)';
    const summaryDevice =
        activeDevice === null || activeDevice === undefined
            ? trainingTargetLabel
            : `${trainingTargetLabel} — ${activeDevice.name}${
                  activeDevice.memory ? `, ${formatBytes(activeDevice.memory)}` : ''
              }`;
    const selectedDatasetName =
        datasets.find((dataset) => dataset.id === selectedDataset?.toString())?.name ?? 'No dataset';
    const selectedPolicyName = MODELS.find((model) => model.id === selectedPolicy)?.name ?? selectedPolicy;

    const goToStep = (offset: number) => {
        const nextStep = steps[currentStepIndex + offset];

        if (nextStep !== undefined) {
            setCurrentStep(nextStep);
        }
    };

    const save = async () => {
        if (isSubmitting) {
            return;
        }

        const dataset_id = selectedDataset?.toString();

        if (!dataset_id || !selectedPolicy || selectedTarget === null) {
            return;
        }

        setIsSubmitting(true);
        setSubmitError(null);
        try {
            if (isRemoteTarget) {
                // Final guard: the remote trainer may have gone offline since the last
                // poll, so re-check availability right before submitting the job.
                const latestHealth = await checkRemoteTrainerHealth();
                if (latestHealth === null || latestHealth.status === 'unreachable') {
                    setSubmitError("Can't reach the remote trainer right now. Make sure it's running, then try again.");
                    return;
                }
            }

            const name = baseModel?.name ?? MODELS.find((policy) => policy.id === selectedPolicy)?.name ?? '';

            const commonPayload = {
                dataset_id,
                project_id: projectId,
                model_name: name,
                policy: selectedPolicy,
                max_epochs: maxEpochs,
                batch_size: batchSize,
                num_workers: numWorkers === 'auto' ? ('auto' as const) : Number(numWorkers),
                auto_scale_batch_size: autoScaleBatchSize,
                precision: (precision?.toString() ?? 'bf16-mixed') as SchemaJob['payload']['precision'],
                compile_model: compileModel,
                // Empty without a mapping step (no fixed camera order, or retraining),
                // which the training side reads as "keep the order it already has".
                image_key_reorder_map: hasFeatureMapping ? featureMapping.imageKeyReorderMap : {},
                num_cameras: hasFeatureMapping ? featureMapping.numCameras : 0,
                // With no formats on offer (the list failed to load, or is empty) there
                // was nothing to choose from, so fall back to exporting every supported
                // format rather than sending [] and exporting none.
                export_backends: exportSelection.backends.length > 0 ? exportSelection.selectedBackends : null,
                lora_enabled: isLoraRequested,
                lora_rank: loraRank,
                lora_alpha: loraAlpha,
                lora_dropout: loraDropout,
                lora_use_dora: isLoraRequested && loraUseDora,
                snapflow_enabled: isSnapflowRequested,
                snapflow_distill_epochs: snapflowDistillEpochs,
                augment_images: augmentImages,
                val_split: 0.1,
                ...extraPayload,
            } as const;

            // Built as an explicit per-branch literal (rather than a single object with
            // a computed `training_target`) so each branch narrows to the matching
            // member of the `SchemaJob['payload']` discriminated union - a computed
            // `training_target` value can't be narrowed to one member by TypeScript.
            const payload: SchemaJob['payload'] = isRemoteTarget
                ? { ...commonPayload, training_target: 'remote', remote_trainer_id: targetRawId(selectedTarget.id) }
                : { ...commonPayload, training_target: 'local' };

            const response = await trainMutation.mutateAsync({ body: payload });
            close(response as SchemaTrainJob | undefined);
        } catch (error) {
            setSubmitError(getApiErrorMessage(error) ?? 'The job could not be submitted. Try again.');
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <Dialog size='L' UNSAFE_style={{ width: 'fit-content' }}>
            <Heading>
                <Flex justifyContent={'space-between'}>
                    <Text> Train model</Text>

                    <TrainingDeviceInfo
                        targetKind={selectedTarget?.kind ?? 'local'}
                        remoteHealth={remoteTrainerHealth ?? null}
                        isCheckingRemote={isCheckingRemoteTrainer}
                    />
                </Flex>
            </Heading>
            <Divider />
            <Content width={'700px'}>
                <Flex direction='column' gap='size-200' width='100%'>
                    {activeStep !== 'setup' && (
                        <TrainingSummaryNote
                            device={summaryDevice}
                            dataset={selectedDatasetName}
                            policy={selectedPolicyName}
                        />
                    )}

                    {submitError !== null && <InlineAlert variant='error'>{submitError}</InlineAlert>}

                    <View minHeight='size-3600'>
                        {activeStep === 'setup' && (
                            <SetupStep
                                datasets={datasets}
                                selectedDataset={selectedDataset}
                                onSelectedDatasetChange={setSelectedDataset}
                                trainingTargetOptions={trainingTargetOptions}
                                targetId={targetId}
                                onTargetIdChange={setTargetId}
                                remoteUnavailable={remoteUnavailable}
                                selectedPolicy={selectedPolicy}
                                onSelectedPolicyChange={setSelectedPolicy}
                                isPolicyDisabled={baseModel !== undefined}
                                activeDevice={activeDevice}
                            />
                        )}

                        {activeStep === 'feature-mapping' && (
                            <FeatureMappingStep policy={selectedPolicy} mapping={featureMapping} />
                        )}

                        {activeStep === 'training-parameters' && (
                            <TrainingParameters
                                maxEpochs={maxEpochs}
                                onMaxEpochsChange={setMaxEpochs}
                                batchSize={batchSize}
                                onBatchSizeChange={setBatchSize}
                                numWorkers={numWorkers}
                                onNumWorkersChange={setNumWorkers}
                                autoScaleBatchSize={autoScaleBatchSize}
                                onAutoScaleBatchSizeChange={setAutoScaleBatchSize}
                                precision={precision}
                                onPrecisionChange={setPrecision}
                                compileModel={compileModel}
                                onCompileModelChange={setCompileModel}
                                augmentImages={augmentImages}
                                onAugmentImagesChange={setAugmentImages}
                                isAutoScaleBatchDisabled={activeDevice?.type !== 'cuda'}
                                deviceType={activeDevice?.type}
                                isLoraSupported={isLoraSupported}
                                loraEnabled={loraEnabled}
                                onLoraEnabledChange={setLoraEnabled}
                                loraRank={loraRank}
                                onLoraRankChange={setLoraRank}
                                loraAlpha={loraAlpha}
                                onLoraAlphaChange={setLoraAlpha}
                                loraDropout={loraDropout}
                                onLoraDropoutChange={setLoraDropout}
                                loraUseDora={loraUseDora}
                                onLoraUseDoraChange={setLoraUseDora}
                                isSnapflowSupported={isSnapflowSupported}
                                snapflowEnabled={snapflowEnabled}
                                onSnapflowEnabledChange={setSnapflowEnabled}
                                snapflowDistillEpochs={snapflowDistillEpochs}
                                onSnapflowDistillEpochsChange={setSnapflowDistillEpochs}
                            />
                        )}

                        {activeStep === 'export' && <ExportStep policy={selectedPolicy} selection={exportSelection} />}
                    </View>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)} isDisabled={isSubmitting}>
                    Cancel
                </Button>
                <Button
                    variant='secondary'
                    onPress={() => goToStep(-1)}
                    isDisabled={currentStepIndex === 0 || isSubmitting}
                >
                    Back
                </Button>
                {isLastStep ? (
                    <Button
                        variant='accent'
                        onPress={save}
                        isPending={isSubmitting}
                        isDisabled={isStepBlocked || isSubmitting}
                    >
                        Train
                    </Button>
                ) : (
                    <Button variant='accent' onPress={() => goToStep(1)} isDisabled={isStepBlocked}>
                        Next
                    </Button>
                )}
            </ButtonGroup>
        </Dialog>
    );
};
