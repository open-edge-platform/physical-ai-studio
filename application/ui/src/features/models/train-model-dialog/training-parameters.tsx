import {
    Checkbox,
    Content,
    ContextualHelp,
    Flex,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    Slider,
    Switch,
    Text,
} from '@geti-ui/ui';

export const MIN_BATCH_SIZE_EXPONENT = 0;
export const MAX_BATCH_SIZE_EXPONENT = 8;

// The slider moves in exponents so every stop is a power of two (1, 2, 4, ... 256),
// which is what the trainer expects; the label maps back to the batch size itself.
export const batchSizeToExponent = (batchSize: number): number => {
    const exponent = Math.round(Math.log2(Math.max(batchSize, 1)));

    return Math.min(Math.max(exponent, MIN_BATCH_SIZE_EXPONENT), MAX_BATCH_SIZE_EXPONENT);
};

export const exponentToBatchSize = (exponent: number): number => 2 ** exponent;

export const RECOMMENDED_PRECISION: Record<string, string> = {
    cuda: 'bf16-mixed',
};

/**
 * Distillation needs at least one epoch to train the policy it distils, plus
 * one to distil it, so a single-epoch run cannot use SnapFlow at all.
 */
export const MIN_EPOCHS_FOR_SNAPFLOW = 2;

export const PRECISION_LABELS: Record<string, string> = {
    'bf16-mixed': 'BF16 Mixed',
    'bf16-true': 'BF16 True',
    '32-true': '32-bit',
};

interface TrainingParametersProps {
    maxEpochs: number;
    onMaxEpochsChange: (value: number) => void;
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
    /** False for policies without a flow-matching sampler; hides the SnapFlow controls entirely. */
    isSnapflowSupported: boolean;
    snapflowEnabled: boolean;
    onSnapflowEnabledChange: (value: boolean) => void;
    snapflowDistillEpochs: number;
    onSnapflowDistillEpochsChange: (value: number) => void;
}

export const TrainingParameters = ({
    maxEpochs,
    onMaxEpochsChange,
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
    isSnapflowSupported,
    snapflowEnabled,
    onSnapflowEnabledChange,
    snapflowDistillEpochs,
    onSnapflowDistillEpochsChange,
}: TrainingParametersProps) => (
    <Flex direction='column' gap='size-150' width='100%'>
        <Flex direction='row' gap='size-150' width='100%'>
            <Flex direction='row' gap='size-300' width='100%'>
                <Slider
                    label='Batch Size'
                    value={batchSizeToExponent(batchSize)}
                    onChange={(exponent) => onBatchSizeChange(exponentToBatchSize(exponent))}
                    getValueLabel={(exponent) => `${exponentToBatchSize(exponent)}`}
                    minValue={MIN_BATCH_SIZE_EXPONENT}
                    maxValue={MAX_BATCH_SIZE_EXPONENT}
                    step={1}
                    flex
                    isFilled
                    isDisabled={autoScaleBatchSize}
                    contextualHelp={
                        <ContextualHelp variant='info'>
                            <Heading>Batch size</Heading>
                            <Content>
                                <Text>
                                    Number of samples processed in each training step. Larger batches use the device
                                    more efficiently and give smoother gradient updates, but need more memory and a much
                                    larger batch may need a higher learning rate to train as well.
                                </Text>
                            </Content>
                        </ContextualHelp>
                    }
                />
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <Switch
                        isEmphasized
                        isSelected={autoScaleBatchSize}
                        onChange={onAutoScaleBatchSizeChange}
                        isDisabled={isAutoScaleBatchDisabled}
                    >
                        Auto
                    </Switch>
                    <ContextualHelp variant='info'>
                        <Heading>Auto scale batch size</Heading>
                        <Content>
                            <Text>
                                Automatically finds the largest batch size that fits in GPU memory before training
                                starts, so the batch size slider is ignored. On XPU auto batch size is disabled.
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
            </Flex>
        </Flex>
        <Flex direction='row' gap='size-150' width='100%'>
            <NumberField
                label='Max Epochs'
                value={maxEpochs}
                onChange={onMaxEpochsChange}
                minValue={1}
                maxValue={1000}
                step={1}
                width='100%'
                contextualHelp={
                    <ContextualHelp variant='info'>
                        <Heading>Max epochs</Heading>
                        <Content>
                            <Text>
                                Total number of training epochs. Training will stop after this many full passes through
                                the dataset. We recommend training for 5 to 10 epochs
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
        <Flex direction='row' gap='size-200' width='100%'>
            <Picker
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
        </Flex>
        <Flex direction='row' gap='size-200' width='100%'>
            <Checkbox isEmphasized isSelected={compileModel} onChange={onCompileModelChange}>
                Compile model
            </Checkbox>
            <ContextualHelp variant='info'>
                <Heading>Compile model</Heading>
                <Content>
                    <Text>
                        Enables torch.compile for all policies. Can significantly speed up training after an initial
                        compilation warmup, but increases startup time.
                    </Text>
                </Content>
            </ContextualHelp>
        </Flex>
        {isSnapflowSupported && (
            <Flex direction='row' gap='size-150' width='100%' alignItems='end'>
                <Flex direction='column' gap='size-150' width='100%' justifyContent='center'>
                    <Flex direction='row' gap='size-100' alignItems='center'>
                        {/* A single-epoch run has no epoch left over to train the policy that gets distilled. */}
                        <Checkbox
                            isEmphasized
                            isSelected={snapflowEnabled}
                            onChange={onSnapflowEnabledChange}
                            isDisabled={maxEpochs < MIN_EPOCHS_FOR_SNAPFLOW}
                        >
                            SnapFlow distillation
                        </Checkbox>
                        <ContextualHelp variant='info'>
                            <Heading>SnapFlow distillation</Heading>
                            <Content>
                                <Text>
                                    Trains normally for the full max epochs, then spends additional epochs distilling
                                    the policy so it generates an action chunk in a single denoising step instead of
                                    ten. The exported model is the distilled one, which runs several times faster on the
                                    robot at comparable task success.
                                </Text>
                            </Content>
                        </ContextualHelp>
                    </Flex>
                </Flex>
                <NumberField
                    label='Distillation epochs'
                    value={snapflowDistillEpochs}
                    onChange={onSnapflowDistillEpochsChange}
                    minValue={1}
                    maxValue={10_000}
                    step={1}
                    width='100%'
                    isDisabled={!snapflowEnabled || maxEpochs < MIN_EPOCHS_FOR_SNAPFLOW}
                    contextualHelp={
                        <ContextualHelp variant='info'>
                            <Heading>Distillation epochs</Heading>
                            <Content>
                                <Text>
                                    How many additional epochs to perform distilling after normal finetuning. We
                                    recommend 2-5 epochs.
                                </Text>
                            </Content>
                        </ContextualHelp>
                    }
                />
            </Flex>
        )}
    </Flex>
);
