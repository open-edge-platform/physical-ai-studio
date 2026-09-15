import { Checkbox, Content, ContextualHelp, Flex, Heading, Item, Key, NumberField, Picker, Text } from '@geti-ui/ui';

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
    /** False for policies without a LoRA/DoRA mixin; hides the LoRA controls entirely. */
    isLoraSupported: boolean;
    loraEnabled: boolean;
    onLoraEnabledChange: (value: boolean) => void;
    loraRank: number;
    onLoraRankChange: (value: number) => void;
    loraAlpha: number | null;
    onLoraAlphaChange: (value: number | null) => void;
    loraDropout: number;
    onLoraDropoutChange: (value: number) => void;
    loraUseDora: boolean;
    onLoraUseDoraChange: (value: boolean) => void;
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
    isLoraSupported,
    loraEnabled,
    onLoraEnabledChange,
    loraRank,
    onLoraRankChange,
    loraAlpha,
    onLoraAlphaChange,
    loraDropout,
    onLoraDropoutChange,
    loraUseDora,
    onLoraUseDoraChange,
    isSnapflowSupported,
    snapflowEnabled,
    onSnapflowEnabledChange,
    snapflowDistillEpochs,
    onSnapflowDistillEpochsChange,
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
                        isEmphasized
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
            <Flex direction='row' alignSelf={'end'} alignItems='center'>
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
        </Flex>
        {isLoraSupported && (
            <Flex direction='column' gap='size-150' width='100%'>
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <Checkbox isEmphasized isSelected={loraEnabled} onChange={onLoraEnabledChange}>
                        LoRA fine-tuning
                    </Checkbox>
                    <ContextualHelp variant='info'>
                        <Heading>LoRA fine-tuning</Heading>
                        <Content>
                            <Text>
                                Freezes the base model and trains small low-rank adapters instead of every parameter.
                                Uses far less memory and trains faster, at the cost of some capacity versus full
                                fine-tuning. The learning rate is automatically scaled up to suit adapter training.
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
                {loraEnabled && (
                    <Flex direction='row' gap='size-150' width='100%'>
                        <NumberField
                            label='LoRA rank'
                            value={loraRank}
                            onChange={onLoraRankChange}
                            minValue={8}
                            maxValue={256}
                            step={8}
                            width='100%'
                            contextualHelp={
                                <ContextualHelp variant='info'>
                                    <Heading>LoRA rank</Heading>
                                    <Content>
                                        <Text>
                                            Dimension of the low-rank decomposition. Higher rank means more trainable
                                            parameters and closer to full fine-tuning. 16 is lighter, 32 is a reasonable
                                            default, 64 gives more capacity for larger datasets.
                                        </Text>
                                    </Content>
                                </ContextualHelp>
                            }
                        />
                        <NumberField
                            label='LoRA alpha'
                            value={loraAlpha ?? undefined}
                            onChange={onLoraAlphaChange}
                            minValue={1}
                            width='100%'
                            contextualHelp={
                                <ContextualHelp variant='info'>
                                    <Heading>LoRA alpha</Heading>
                                    <Content>
                                        <Text>
                                            Scaling numerator (scaling = alpha / rank). Leave empty to default to the
                                            rank (scaling = 1.0). Increase for a stronger adaptation signal.
                                        </Text>
                                    </Content>
                                </ContextualHelp>
                            }
                        />
                        <NumberField
                            label='LoRA dropout'
                            value={loraDropout}
                            onChange={onLoraDropoutChange}
                            minValue={0}
                            maxValue={0.99}
                            step={0.01}
                            width='100%'
                            contextualHelp={
                                <ContextualHelp variant='info'>
                                    <Heading>LoRA dropout</Heading>
                                    <Content>
                                        <Text>Dropout probability applied to LoRA adapter inputs.</Text>
                                    </Content>
                                </ContextualHelp>
                            }
                        />
                        <Flex direction='column' gap='size-150' width='100%' justifyContent='center'>
                            <Flex direction='row' gap='size-100' alignItems='center'>
                                <Checkbox isEmphasized isSelected={loraUseDora} onChange={onLoraUseDoraChange}>
                                    Use DoRA
                                </Checkbox>
                                <ContextualHelp variant='info'>
                                    <Heading>Use DoRA</Heading>
                                    <Content>
                                        <Text>
                                            Weight-Decomposed Low-Rank Adaptation: learns a per-column magnitude vector
                                            on top of the LoRA update. Typically improves quality at low ranks at the
                                            cost of slightly more compute/memory.
                                        </Text>
                                    </Content>
                                </ContextualHelp>
                            </Flex>
                        </Flex>
                    </Flex>
                )}
            </Flex>
        )}
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
