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
    Text,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaTrainJob as SchemaJob, SchemaModel } from '../../../api/openapi-spec';
import { useProject } from '../../projects/use-project';
import { useRemoteTrainerHealth } from '../../remote-trainers/use-remote-trainer-health';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
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

type TrainingTargetOption = {
    id: string;
    label: string;
};

export const TrainModelDialog = ({ baseModel, close, defaultMaxEpochs = 5 }: TrainModelDialogProps) => {
    const bestDevice = useBestTrainingDevice();
    const { data: remoteTrainers = [] } = $api.useQuery('get', '/api/remote-trainers');
    // Continuing an existing model needs its checkpoint, which only this machine
    // has: the trainer protocol can receive a dataset but not a base checkpoint.
    // So a resumed run offers local training only.
    const canTrainRemotely = baseModel === undefined;
    const trainingTargetOptions: TrainingTargetOption[] = [
        { id: 'local', label: 'This machine (local)' },
        ...(canTrainRemotely
            ? remoteTrainers.map((remoteTrainer) => ({
                  id: remoteTrainer.id,
                  label: remoteTrainer.name,
              }))
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
    const [remoteTrainerId, setRemoteTrainerId] = useState<Key | null>('local');
    const isRemoteTarget = remoteTrainerId !== null && remoteTrainerId !== 'local';
    const {
        health: remoteTrainerHealth,
        isChecking: isCheckingRemoteTrainer,
        checkHealth: checkRemoteTrainerHealth,
    } = useRemoteTrainerHealth(isRemoteTarget ? (remoteTrainerId?.toString() ?? null) : null);
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
    // The device actually driving this job: the local GPU when training locally,
    // or the remote trainer's reported GPU once its health check resolves. Auto
    // scale/precision defaults and the disabled state below should track whichever
    // one is currently in play, the same way they did when there was only ever a
    // single active device to consider.
    const activeDevice = isRemoteTarget ? bestRemoteDevice : bestDevice;

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

        if (!dataset_id || !selectedPolicy || remoteTrainerId === null) {
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
            training_target: isRemoteTarget ? 'remote' : 'local',
            ...(isRemoteTarget ? { remote_trainer_id: remoteTrainerId?.toString() } : {}),
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
                        isRemoteTarget={isRemoteTarget}
                        remoteHealth={remoteTrainerHealth ?? null}
                        isCheckingRemote={isCheckingRemoteTrainer}
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
                        selectedKey={remoteTrainerId}
                        onSelectionChange={setRemoteTrainerId}
                        width='100%'
                        items={trainingTargetOptions}
                    >
                        {(trainingTarget) => <Item key={trainingTarget.id}>{trainingTarget.label}</Item>}
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
                        remoteTrainerId === null ||
                        remoteUnavailable ||
                        policyAccessBlocksTraining
                    }
                >
                    Train
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
