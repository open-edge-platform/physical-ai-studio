import { useMemo, useState } from 'react';

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
    Form,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    StatusLight,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';

import { $api } from '../../api/client';
import { SchemaDeviceInfo, SchemaTrainJob as SchemaJob, SchemaModel } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { InlineAlert } from '../../features/robots/setup-wizard/shared/inline-alert';

import classes from './train-model-dialog.module.scss';

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
        id: 'pi0',
        name: 'Pi0',
        description: 'Vision-Language-Action model based on PaliGemma 3B',
        minVRAM: 12 * GB,
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

const useBestTrainingDevice = (): SchemaDeviceInfo | null => {
    const { data: trainingDevices = [] } = $api.useQuery('get', '/api/system/devices/training');

    // Pick the GPU with the most VRAM (if any)
    return useMemo(() => {
        return trainingDevices
            .filter((d) => d.type !== 'cpu' && d.memory != null)
            .reduce((best, device): SchemaDeviceInfo | null => {
                if (best === null || (device.memory ?? 0) > (best.memory ?? 0)) {
                    return device;
                }

                return best;
            }, null);
    }, [trainingDevices]);
};

const TrainingDeviceInfo = () => {
    const bestDevice = useBestTrainingDevice();

    return (
        <Flex UNSAFE_style={{ textAlign: 'right' }} direction='column' gap='size-75'>
            {bestDevice ? (
                <StatusLight variant='positive'>
                    {bestDevice.name}, {formatBytes(bestDevice.memory!)} VRAM
                </StatusLight>
            ) : (
                <StatusLight variant='neutral'>CPU only (no GPU detected)</StatusLight>
            )}
        </Flex>
    );
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
                    isDisabled={autoScaleBatchSize || isAutoScaleBatchDisabled}
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
                selectedKey={precision}
                onSelectionChange={onPrecisionChange}
                contextualHelp={
                    <ContextualHelp variant='info'>
                        <Heading>Training precision</Heading>
                        <Content>
                            <Text>
                                Controls numerical precision during training. Default selects the optimal precision for
                                your device (BF16 Mixed for CUDA, 32-bit for XPU). BF16 Mixed uses half-precision where
                                safe for faster training and lower memory usage. 32-bit uses full precision for maximum
                                numerical stability.
                            </Text>
                        </Content>
                    </ContextualHelp>
                }
            >
                <Item key='default'>Default</Item>
                <Item key='bf16-mixed'>BF16 Mixed</Item>
                <Item key='32'>32-bit</Item>
            </Picker>
            <Flex direction='column' gap='size-150' width='100%' justifyContent='end'>
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <Checkbox isSelected={compileModel} onChange={onCompileModelChange}>
                        Compile model
                    </Checkbox>
                    <ContextualHelp variant='info'>
                        <Heading>Compile model</Heading>
                        <Content>
                            <Text>
                                Enables torch.compile for all policies. Can significantly speed up training after an
                                initial compilation warmup, but increases startup time. If compilation fails, training
                                automatically retries without compilation.
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
            </Flex>
        </Flex>
);

export const TrainModelDialog = ({ baseModel, close, defaultMaxSteps = 10000 }: TrainModelDialogProps) => {
    const bestDevice = useBestTrainingDevice();

    const defaultName = baseModel?.name ?? '';
    const defaultDatasetId = baseModel?.dataset_id ?? null;
    const extraPayload = baseModel ? { base_model_id: baseModel.id! } : undefined;

    const [selectedPolicy, setSelectedPolicy] = useState<string>(baseModel?.policy ?? 'act');
    const { datasets, id: projectId } = useProject();

    const [name, setName] = useState<string>(defaultName);
    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxSteps, setMaxSteps] = useState<number>(defaultMaxSteps);
    const [batchSize, setBatchSize] = useState<number>(8);
    const [numWorkers, setNumWorkers] = useState<Key | null>('auto');
    const [autoScaleBatchSize, setAutoScaleBatchSize] = useState<boolean>(bestDevice?.type === 'cuda');
    const [precision, setPrecision] = useState<Key | null>('default');
    const [compileModel, setCompileModel] = useState<boolean>(false);

    const trainMutation = $api.useMutation('post', '/api/jobs:train', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
    });

    const save = () => {
        const dataset_id = selectedDataset?.toString();

        if (!dataset_id || !selectedPolicy) {
            return;
        }

        const payload: SchemaJob['payload'] = {
            dataset_id,
            project_id: projectId,
            model_name: name,
            policy: selectedPolicy,
            max_steps: maxSteps,
            batch_size: batchSize,
            num_workers: numWorkers === 'auto' ? 'auto' : Number(numWorkers),
            auto_scale_batch_size: autoScaleBatchSize,
            precision: precision?.toString() ?? 'default',
            compile_model: compileModel,
            val_split: 0.1,
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

                    <TrainingDeviceInfo />
                </Flex>
            </Heading>
            <Divider />
            <Content width={'700px'}>
                <Form
                    onSubmit={(e) => {
                        e.preventDefault();
                        save();
                    }}
                    validationBehavior='native'
                >
                    <Flex direction='column' gap='size-200' width='100%'>
                        <TextField label='Name' value={name} onChange={setName} width='100%' />

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

                        <PolicySelection
                            selectedPolicy={selectedPolicy}
                            onSelectionChange={setSelectedPolicy}
                            isDisabled={baseModel !== undefined}
                            trainingDevice={bestDevice}
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
                                    isAutoScaleBatchDisabled={bestDevice?.type !== 'cuda'}
                                />
                            </DisclosurePanel>
                        </Disclosure>
                    </Flex>
                </Form>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={save}>
                    Train
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
