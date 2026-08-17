import { useEffect, useMemo, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Flex,
    Heading,
    Item,
    Key,
    Picker,
    StatusLight,
    Text,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaTrainJob as SchemaJob, SchemaModel } from '../../../api/openapi-spec';
import { useProject } from '../../projects/use-project';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import {
    remoteServerComputeDetail,
    remoteServerStatusLabel,
    remoteServerStatusVariant,
} from '../../training-targets/remote-server-status-utils';
import { getDisplayHealth, healthLabel, healthVariant } from '../../training-targets/remote-trainer-health-utils';
import { useRemoteServersStatus } from '../../training-targets/training-targets-table/use-remote-servers-status';
import { useRemoteTrainersHealth } from '../../training-targets/training-targets-table/use-remote-trainers-health';
import { useRemoteTrainerHealth } from '../../training-targets/use-remote-trainer-health';
import { MODELS } from './policies';
import { PolicyAccessAlert } from './policy-access-alert';
import { PolicySelection } from './policy-selection';
import { TrainingDeviceInfo } from './training-device-info';
import { TrainingParameters } from './training-parameters';
import { pickBestDevice, useBestTrainingDevice } from './use-training-devices';

import classes from './train-model-dialog.module.css';

export type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaJob['payload'];
};

interface TrainModelDialogProps {
    baseModel?: SchemaModel;
    close: (job: SchemaJob | undefined) => void;
    defaultMaxEpochs?: number;
}

export type TrainingTargetKind = 'local' | 'trainer' | 'ssh';

type TrainingTargetStatusVariant = 'positive' | 'notice' | 'negative' | 'neutral' | 'yellow';

type TrainingTargetOption = {
    id: string;
    label: string;
    /** `local`, `trainer:<remote_trainer_id>`, or `ssh:<remote_server_id>`. */
    kind: TrainingTargetKind;
    statusVariant: TrainingTargetStatusVariant;
    statusLabel: string;
};

const LOCAL_TARGET_ID = 'local';

/** Strip the `trainer:`/`ssh:` prefix off a training-target option id. */
const targetRawId = (id: string): string => id.split(':', 2)[1] ?? id;

export const TrainModelDialog = ({ baseModel, close, defaultMaxEpochs = 5 }: TrainModelDialogProps) => {
    const bestDevice = useBestTrainingDevice();
    const { data: remoteTrainers = [] } = $api.useQuery('get', '/api/remote-trainers');
    const { data: remoteServers = [] } = $api.useQuery('get', '/api/remote-servers');
    // Continuing an existing model needs its checkpoint, which only this machine
    // has: the trainer protocol can receive a dataset but not a base checkpoint.
    // So a resumed run offers local training only.
    const canTrainRemotely = baseModel === undefined;
    const remoteTrainerHealthById = useRemoteTrainersHealth(canTrainRemotely ? remoteTrainers.map((t) => t.id) : []);
    const remoteServerStatusById = useRemoteServersStatus(canTrainRemotely ? remoteServers.map((s) => s.id) : []);
    // One control lists every target type (local, direct-URL trainer, SSH
    // server) rather than a separate remote-server dropdown or a local/remote
    // mode toggle, so the derived `training_target` is always unambiguous.
    // Each option carries its own status variant/label so the "Run on" dropdown
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
        ...(canTrainRemotely
            ? remoteServers.map((remoteServer) => {
                  const entry = remoteServerStatusById.get(remoteServer.id);
                  const isChecking = entry?.isChecking ?? false;
                  const isVerified = remoteServer.last_check_status === 'healthy';
                  return {
                      id: `ssh:${remoteServer.id}`,
                      label: remoteServer.name,
                      kind: 'ssh' as const,
                      statusVariant: isVerified
                          ? remoteServerStatusVariant(entry?.status, isChecking)
                          : ('negative' as const),
                      statusLabel: isVerified ? remoteServerStatusLabel(entry?.status, isChecking) : 'Not verified',
                  };
              })
            : []),
    ];

    const defaultDatasetId = baseModel?.dataset_id ?? null;
    const extraPayload = baseModel ? { base_model_id: baseModel.id! } : undefined;

    const [selectedPolicy, setSelectedPolicy] = useState<string>(baseModel?.policy ?? 'act');
    const { datasets, id: projectId } = useProject();

    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxEpochs, setMaxEpochs] = useState<number>(defaultMaxEpochs);
    const [batchSize, setBatchSize] = useState<number>(8);
    const [numWorkers, setNumWorkers] = useState<Key | null>('auto');
    const [autoScaleBatchSize, setAutoScaleBatchSize] = useState<boolean>(bestDevice?.type === 'cuda');
    const [precision, setPrecision] = useState<Key | null>(bestDevice?.type === 'cuda' ? 'bf16-mixed' : '32-true');
    const [compileModel, setCompileModel] = useState<boolean>(false);
    const [targetId, setTargetId] = useState<Key | null>(LOCAL_TARGET_ID);
    const selectedTarget = trainingTargetOptions.find((option) => option.id === targetId) ?? null;
    const isRemoteTarget = selectedTarget?.kind === 'trainer';
    const isSshTarget = selectedTarget?.kind === 'ssh';
    const {
        health: remoteTrainerHealth,
        isChecking: isCheckingRemoteTrainer,
        checkHealth: checkRemoteTrainerHealth,
    } = useRemoteTrainerHealth(isRemoteTarget ? targetRawId(selectedTarget.id) : null);
    const remoteUnavailable = isRemoteTarget && remoteTrainerHealth?.status === 'unreachable';
    const { data: policyAccess, isLoading: isCheckingPolicyAccess } = $api.useQuery(
        'get',
        '/api/policies/{policy}/huggingface-access',
        {
            params: { path: { policy: selectedPolicy } },
        }
    );
    const policyAccessBlocksTraining =
        isCheckingPolicyAccess ||
        policyAccess?.requirements.some(
            (requirement) =>
                requirement.required && (requirement.status === 'missing_token' || requirement.status === 'denied')
        ) === true;
    const bestRemoteDevice = useMemo(() => pickBestDevice(remoteTrainerHealth?.devices ?? []), [remoteTrainerHealth]);
    const selectedSshServer = useMemo(() => {
        if (!isSshTarget) {
            return null;
        }
        const rawId = targetRawId(selectedTarget.id);
        return remoteServers.find((server) => server.id === rawId) ?? null;
    }, [isSshTarget, remoteServers, selectedTarget]);
    // Live Tier-1 status for the selected server, polled independently of the
    // persisted `last_check_status`. A server can be verified (last_check_status
    // === "healthy") yet go unreachable/degraded before the next explicit
    // verification, so gate on both rather than trusting the persisted flag alone.
    const selectedSshStatusEntry = selectedSshServer ? remoteServerStatusById.get(selectedSshServer.id) : undefined;
    const sshVerified = selectedSshServer?.last_check_status === 'healthy';
    const sshLiveStatus = selectedSshStatusEntry?.status?.status;
    const sshUnavailable =
        isSshTarget && (!sshVerified || (sshLiveStatus !== undefined && sshLiveStatus !== 'healthy'));
    // Human-readable reason for the warning banner below, falling back to an
    // explicit label rather than rendering `undefined` when the server isn't
    // found in `remoteServers` yet (e.g. still loading).
    const sshStatusMessage = !selectedSshServer
        ? 'not loaded yet'
        : !sshVerified
          ? selectedSshServer.last_check_status
          : remoteServerStatusLabel(selectedSshStatusEntry?.status, selectedSshStatusEntry?.isChecking ?? false);
    const selectedSshComputeDetail = useMemo(() => {
        if (!isSshTarget || selectedSshServer === null) {
            return undefined;
        }
        return remoteServerComputeDetail(selectedSshStatusEntry?.status);
    }, [isSshTarget, selectedSshServer, selectedSshStatusEntry]);
    // The device actually driving this job: the local GPU when training locally,
    // the remote trainer's reported GPU once its health check resolves, or the
    // configured accelerator for an SSH-provisioned server (no live VRAM probe,
    // since Studio never dials Tier 2 verification from this dialog). Auto
    // scale/precision defaults and the disabled state below should track
    // whichever one is currently in play.
    const activeDevice = useMemo(() => {
        if (isRemoteTarget) {
            return bestRemoteDevice;
        }
        if (isSshTarget && selectedSshServer) {
            return { type: selectedSshServer.device_type, name: selectedSshServer.name };
        }
        return bestDevice;
    }, [isRemoteTarget, isSshTarget, bestRemoteDevice, selectedSshServer, bestDevice]);

    useEffect(() => {
        if (activeDevice?.type === 'cuda') {
            setPrecision('bf16-mixed');
            setAutoScaleBatchSize(true);
        } else {
            setPrecision('32-true');
            setAutoScaleBatchSize(false);
        }
    }, [activeDevice]);

    const trainMutation = $api.useMutation('post', '/api/jobs:train', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
    });

    const save = async () => {
        const dataset_id = selectedDataset?.toString();

        if (!dataset_id || !selectedPolicy || selectedTarget === null) {
            return;
        }

        if (isRemoteTarget) {
            // Final guard: the remote trainer may have gone offline since the last
            // poll, so re-check availability right before submitting the job.
            const latestHealth = await checkRemoteTrainerHealth();
            if (latestHealth === null || latestHealth.status === 'unreachable') {
                return;
            }
        }

        if (isSshTarget) {
            if (sshUnavailable) {
                return;
            }
            // Final guard: the server may have gone unreachable/degraded since the
            // last poll, so re-check its Tier-1 status right before submitting.
            const latestStatus = await selectedSshStatusEntry?.checkStatus();
            if (!latestStatus || latestStatus.status !== 'healthy') {
                return;
            }
        }

        const name = baseModel?.name ?? MODELS.find((policy) => policy.id === selectedPolicy)?.name ?? '';

        const payload: SchemaJob['payload'] = {
            dataset_id,
            project_id: projectId,
            model_name: name,
            policy: selectedPolicy,
            max_epochs: maxEpochs,
            batch_size: batchSize,
            num_workers: numWorkers === 'auto' ? 'auto' : Number(numWorkers),
            auto_scale_batch_size: autoScaleBatchSize,
            precision: (precision?.toString() ?? 'bf16-mixed') as SchemaJob['payload']['precision'],
            compile_model: compileModel,
            val_split: 0.1,
            training_target: isRemoteTarget ? 'remote' : isSshTarget ? 'ssh' : 'local',
            ...(isRemoteTarget ? { remote_trainer_id: targetRawId(selectedTarget.id) } : {}),
            ...(isSshTarget ? { remote_server_id: targetRawId(selectedTarget.id) } : {}),
            ...extraPayload,
        };

        trainMutation.mutateAsync({ body: payload }).then((response) => {
            close(response as SchemaTrainJob | undefined);
        });
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
                        sshServer={selectedSshServer}
                        sshComputeDetail={selectedSshComputeDetail}
                    />
                </Flex>
            </Heading>
            <Divider />
            <Content width={'700px'}>
                <Flex direction='column' gap='size-200' width='100%'>
                    {remoteUnavailable && (
                        <InlineAlert variant='warning'>
                            Can&apos;t reach the remote trainer, so training can&apos;t start. Make sure it&apos;s
                            running, then try again.
                        </InlineAlert>
                    )}

                    {sshUnavailable && (
                        <InlineAlert variant='warning'>
                            This remote server isn&apos;t ready for training (status: {sshStatusMessage}). Verify the
                            server before submitting a job.
                        </InlineAlert>
                    )}

                    <Picker
                        label='Dataset'
                        selectedKey={selectedDataset}
                        onSelectionChange={setSelectedDataset}
                        width='100%'
                    >
                        {datasets.map((dataset) => (
                            <Item key={dataset.id}>{dataset.name}</Item>
                        ))}
                    </Picker>

                    <Picker
                        label='Run on'
                        selectedKey={targetId}
                        onSelectionChange={setTargetId}
                        width='100%'
                        items={trainingTargetOptions}
                    >
                        {(trainingTarget) => (
                            <Item key={trainingTarget.id} textValue={trainingTarget.label}>
                                <Text>{trainingTarget.label}</Text>
                                {/* `Item` only recognizes plain `Text` children for its label/description
                                    slots (a `StatusLight` isn't one), so nest it inside the description
                                    slot rather than passing it as a sibling — otherwise both children
                                    collapse into the same "label" grid area and overlap. */}
                                <Text slot='description'>{trainingTarget.statusLabel}</Text>
                                <Text slot={'icon'}>
                                    <StatusLight variant={trainingTarget.statusVariant} marginBottom={0} />
                                </Text>
                            </Item>
                        )}
                    </Picker>

                    <PolicySelection
                        selectedPolicy={selectedPolicy}
                        onSelectionChange={setSelectedPolicy}
                        isDisabled={baseModel !== undefined}
                        trainingDevice={activeDevice}
                    />
                    <PolicyAccessAlert policy={selectedPolicy} />

                    <Disclosure
                        isQuiet
                        UNSAFE_style={{ padding: 0 }}
                        UNSAFE_className={classes.advancedSettingsDisclosure}
                        defaultExpanded={bestDevice?.type !== 'cuda'}
                    >
                        <DisclosureTitle UNSAFE_style={{ fontSize: 13, padding: '4px 0' }}>
                            Advanced settings
                        </DisclosureTitle>
                        <DisclosurePanel UNSAFE_style={{ padding: 0 }}>
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
                                isAutoScaleBatchDisabled={activeDevice?.type !== 'cuda'}
                                deviceType={activeDevice?.type}
                            />
                        </DisclosurePanel>
                    </Disclosure>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={save}
                    isDisabled={
                        !selectedDataset ||
                        !selectedPolicy ||
                        selectedTarget === null ||
                        remoteUnavailable ||
                        sshUnavailable ||
                        policyAccessBlocksTraining
                    }
                >
                    Train
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
