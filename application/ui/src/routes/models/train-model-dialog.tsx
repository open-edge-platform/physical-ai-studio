import { useEffect, useMemo, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Card,
    Checkbox,
    Content,
    ContextualHelp,
    Dialog,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Flex,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    StatusLight,
    Text,
    View,
} from '@geti-ui/ui';

import { $api } from '../../api/client';
import {
    SchemaDeviceInfo,
    SchemaTrainJob as SchemaJob,
    SchemaModel,
    SchemaRemoteTrainerHealth,
} from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { useRemoteTrainerHealth } from '../../features/remote-trainers/use-remote-trainer-health';
import { InlineAlert } from '../../features/robots/setup-wizard/shared/inline-alert';

import classes from './train-model-dialog.module.css';

export type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaJob['payload'];
};

const GB = 1024 ** 3;

/** Format bytes as a human-readable GB string. */
const formatBytes = (bytes: number): string => {
    const gb = bytes / GB;
    return gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
};

/**
 * Available training policies with hardware requirements.
 *
 * `minVRAM` is the estimated minimum VRAM (in bytes) required to train with batch_size=1.
 */
export const MODELS: ReadonlyArray<{
    id: string;
    name: string;
    description: string;
    minVRAM: number;
}> = [
    {
        id: 'act',
        name: 'ACT',
        description: 'Action Chunking with Transformers, lightweight and fast to train',
        minVRAM: 2 * GB,
    },
    {
        id: 'smolvla',
        name: 'SmolVLA',
        description: 'Small Vision-Language-Action model based on SmolVLM2-500M',
        minVRAM: 8 * GB,
    },
    {
        id: 'pi05',
        name: 'Pi0.5',
        description: 'Enhanced Pi0 with discrete state encoding and longer context',
        minVRAM: 16 * GB,
    },
];

interface TrainModelDialogProps {
    baseModel?: SchemaModel;
    close: (job: SchemaJob | undefined) => void;
    defaultMaxSteps?: number;
}

type TrainingTargetOption = {
    id: string;
    label: string;
};

interface PolicySelectionProps {
    selectedPolicy: string;
    onSelectionChange: (policy: string) => void;
    isDisabled?: boolean;
    trainingDevice: SchemaDeviceInfo | null;
}

const PolicySelection = ({ selectedPolicy, onSelectionChange, isDisabled, trainingDevice }: PolicySelectionProps) => {
    const availableVram = trainingDevice?.memory ?? 0;

    const selectedModel = MODELS.find((m) => m.id === selectedPolicy) ?? null;
    const hasInsufficientVram = selectedModel !== null && availableVram > 0 && selectedModel.minVRAM > availableVram;

    return (
        <Flex direction='column' gap='size-100'>
            <Text UNSAFE_style={{ fontSize: 12 }}>Policy</Text>
            <div className={classes.policyGrid}>
                {MODELS.map((model) => {
                    const isSelected = selectedPolicy === model.id;
                    if (isDisabled && !isSelected) {
                        return null;
                    }

                    return (
                        <Card
                            key={model.id}
                            aria-label={`Select ${model.name} policy`}
                            isSelected={isSelected}
                            isDisabled={isDisabled}
                            onPress={() => onSelectionChange(model.id)}
                            UNSAFE_className={classes.modelPolicyCard}
                        >
                            <Flex direction='column' gap='size-100'>
                                <Flex justifyContent={'space-between'}>
                                    <Text
                                        UNSAFE_style={{
                                            fontWeight: 700,
                                            color: selectedPolicy === model.id ? 'var(--energy-blue)' : undefined,
                                        }}
                                    >
                                        {model.name}
                                    </Text>
                                    <Flex
                                        UNSAFE_style={{ fontSize: 11, opacity: 0.7, textAlign: 'right' }}
                                        direction='column'
                                        gap='size-50'
                                    >
                                        <Text>&ge; {formatBytes(model.minVRAM)} VRAM</Text>
                                    </Flex>
                                </Flex>
                                <Divider size='S' />
                                <Text UNSAFE_style={{ fontSize: 12 }}>{model.description}</Text>
                            </Flex>
                        </Card>
                    );
                })}
            </div>

            {hasInsufficientVram && (
                <View marginTop='size-100'>
                    <InlineAlert variant='warning'>
                        {selectedModel.name} requires at least {formatBytes(selectedModel!.minVRAM)} VRAM but your
                        device has {formatBytes(availableVram)}. Training may fail or be very slow.
                    </InlineAlert>
                </View>
            )}
        </Flex>
    );
};

/** Pick the device with the most VRAM (if any) from a list of reported devices. */
const pickBestDevice = (devices: SchemaDeviceInfo[]): SchemaDeviceInfo | null =>
    devices
        .filter((d) => d.type !== 'cpu' && d.memory != null)
        .reduce((best: SchemaDeviceInfo | null, device) => {
            if (best === null || (device.memory ?? 0) > (best.memory ?? 0)) {
                return device;
            }

            return best;
        }, null);

const useBestTrainingDevice = (): SchemaDeviceInfo | null => {
    const { devices } = useTrainingDevices();

    return useMemo(() => pickBestDevice(devices), [devices]);
};

/**
 * Reads the training devices endpoint and normalizes the response.
 *
 * The endpoint always reports this Studio host's local training devices. Remote
 * trainer configuration is selected independently for each submitted job.
 */
const useTrainingDevices = () => {
    const { data } = $api.useQuery('get', '/api/system/devices/training', {}, { refetchOnMount: 'always' });

    return {
        devices: data?.devices ?? [],
    };
};

interface TrainingDeviceInfoProps {
    isRemoteTarget: boolean;
    remoteHealth: SchemaRemoteTrainerHealth | null;
    isCheckingRemote: boolean;
}

const TrainingDeviceInfo = ({ isRemoteTarget, remoteHealth, isCheckingRemote }: TrainingDeviceInfoProps) => {
    const bestDevice = useBestTrainingDevice();
    const bestRemoteDevice = useMemo(() => pickBestDevice(remoteHealth?.devices ?? []), [remoteHealth]);

    return (
        <Flex UNSAFE_style={{ textAlign: 'right' }} direction='column' gap='size-75'>
            {isRemoteTarget ? (
                remoteHealth?.status === 'unreachable' ? (
                    <StatusLight variant='negative'>Remote trainer unavailable</StatusLight>
                ) : bestRemoteDevice ? (
                    <StatusLight variant='positive'>
                        {bestRemoteDevice.name}, {formatBytes(bestRemoteDevice.memory!)} VRAM
                    </StatusLight>
                ) : isCheckingRemote && remoteHealth === null ? (
                    <StatusLight variant='neutral'>Checking remote trainer…</StatusLight>
                ) : (
                    <StatusLight variant='neutral'>Remote trainer selected</StatusLight>
                )
            ) : bestDevice ? (
                <StatusLight variant='positive'>
                    {bestDevice.name}, {formatBytes(bestDevice.memory!)} VRAM
                </StatusLight>
            ) : (
                <StatusLight variant='neutral'>CPU only (no GPU detected)</StatusLight>
            )}
        </Flex>
    );
};

const RECOMMENDED_PRECISION: Record<string, string> = {
    cuda: 'bf16-mixed',
};

const PRECISION_LABELS: Record<string, string> = {
    'bf16-mixed': 'BF16 Mixed',
    'bf16-true': 'BF16 True',
    '32-true': '32-bit',
};

interface TrainingParametersProps {
    maxSteps: number;
    onMaxStepsChange: (value: number) => void;
    batchSize: number;
    onBatchSizeChange: (value: number) => void;
    numWorkers: Key | null;
    onNumWorkersChange: (value: Key | null) => void;
    autoScaleBatchSize: boolean;
    onAutoScaleBatchSizeChange: (value: boolean) => void;
    precision: Key | null;
    onPrecisionChange: (value: Key | null) => void;
    compileModel: boolean;
    onCompileModelChange: (value: boolean) => void;
    isAutoScaleBatchDisabled: boolean;
    deviceType: string | undefined;
}

const TrainingParameters = ({
    maxSteps,
    onMaxStepsChange,
    batchSize,
    onBatchSizeChange,
    numWorkers,
    onNumWorkersChange,
    autoScaleBatchSize,
    onAutoScaleBatchSizeChange,
    precision,
    onPrecisionChange,
    compileModel,
    onCompileModelChange,
    isAutoScaleBatchDisabled,
    deviceType,
}: TrainingParametersProps) => (
    <Flex direction='column' gap='size-150' width='100%'>
        <Flex direction='row' gap='size-150' width='100%'>
            <Flex direction='column' gap='size-150' width='100%'>
                <NumberField
                    label='Batch Size'
                    value={batchSize}
                    onChange={onBatchSizeChange}
                    minValue={1}
                    maxValue={256}
                    step={1}
                    width='100%'
                    isDisabled={autoScaleBatchSize}
                    flex
                />
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <Checkbox
                        isSelected={autoScaleBatchSize}
                        onChange={onAutoScaleBatchSizeChange}
                        isDisabled={isAutoScaleBatchDisabled}
                    >
                        Auto scale batch size
                    </Checkbox>
                    <ContextualHelp variant='info'>
                        <Heading>Auto scale batch size</Heading>
                        <Content>
                            <Text>
                                Automatically finds the largest batch size that fits in GPU memory before training
                                starts. On XPU auto batch size is disabled.
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
            </Flex>
            <NumberField
                label='Max Steps'
                value={maxSteps}
                onChange={onMaxStepsChange}
                minValue={100}
                maxValue={100000}
                step={100}
                width='100%'
                contextualHelp={
                    <ContextualHelp variant='info'>
                        <Heading>Max steps</Heading>
                        <Content>
                            <Text>
                                Total number of gradient update steps. Training will stop after this many steps
                                regardless of epochs.
                            </Text>
                        </Content>
                    </ContextualHelp>
                }
            />
            <Picker
                width='100%'
                label='Data Workers'
                selectedKey={numWorkers}
                onSelectionChange={onNumWorkersChange}
                contextualHelp={
                    <ContextualHelp variant='info'>
                        <Heading>Data workers</Heading>
                        <Content>
                            <Text>
                                Number of parallel processes for loading training data. Auto selects a value based on
                                available CPU cores. More workers can speed up training but use more memory.
                            </Text>
                        </Content>
                    </ContextualHelp>
                }
            >
                <Item key='auto'>Auto</Item>
                <Item key='0'>0 (main process)</Item>
                <Item key='1'>1</Item>
                <Item key='2'>2</Item>
                <Item key='4'>4</Item>
                <Item key='8'>8</Item>
                <Item key='16'>16</Item>
            </Picker>
        </Flex>
        <Flex direction='row' gap='size-150' width='100%'>
            <Picker
                width='100%'
                label='Precision'
                description={
                    deviceType
                        ? `${
                              PRECISION_LABELS[RECOMMENDED_PRECISION[deviceType] ?? '32-true']
                          } recommended for ${deviceType.toUpperCase()}`
                        : undefined
                }
                selectedKey={precision}
                onSelectionChange={onPrecisionChange}
                contextualHelp={
                    <ContextualHelp variant='info'>
                        <Heading>Training precision</Heading>
                        <Content>
                            <Text>
                                Controls numerical precision during training. BF16 Mixed uses half-precision where safe
                                for faster training and lower memory usage. BF16 True runs entirely in BF16 for maximum
                                speed. 32-bit uses full precision for maximum numerical stability.
                            </Text>
                        </Content>
                    </ContextualHelp>
                }
            >
                <Item key='bf16-mixed'>BF16 Mixed</Item>
                <Item key='bf16-true'>BF16 True</Item>
                <Item key='32-true'>32-bit</Item>
            </Picker>
            <Flex direction='column' gap='size-150' width='100%' justifyContent='center'>
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <Checkbox isSelected={compileModel} onChange={onCompileModelChange}>
                        Compile model
                    </Checkbox>
                    <ContextualHelp variant='info'>
                        <Heading>Compile model</Heading>
                        <Content>
                            <Text>
                                Enables torch.compile for all policies. Can significantly speed up training after an
                                initial compilation warmup, but increases startup time.
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
            </Flex>
        </Flex>
    </Flex>
);

export const TrainModelDialog = ({ baseModel, close, defaultMaxSteps = 10000 }: TrainModelDialogProps) => {
    const bestDevice = useBestTrainingDevice();
    const { data: remoteTrainers = [] } = $api.useQuery('get', '/api/remote-trainers');
    const trainingTargetOptions: TrainingTargetOption[] = [
        { id: 'local', label: 'This machine (local)' },
        ...remoteTrainers.map((remoteTrainer) => ({
            id: remoteTrainer.id,
            label: remoteTrainer.name,
        })),
    ];

    const defaultDatasetId = baseModel?.dataset_id ?? null;
    const extraPayload = baseModel ? { base_model_id: baseModel.id! } : undefined;

    const [selectedPolicy, setSelectedPolicy] = useState<string>(baseModel?.policy ?? 'act');
    const { datasets, id: projectId } = useProject();

    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxSteps, setMaxSteps] = useState<number>(defaultMaxSteps);
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
            max_steps: maxSteps,
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
                                maxSteps={maxSteps}
                                onMaxStepsChange={setMaxSteps}
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
                    isDisabled={!selectedDataset || !selectedPolicy || remoteTrainerId === null || remoteUnavailable}
                >
                    Train
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
