import { Card, Divider, Flex, Text, View } from '@geti-ui/ui';

import { SchemaDeviceInfo } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { formatBytes, MODELS } from './policies';

import classes from './train-model-dialog.module.css';

interface PolicySelectionProps {
    selectedPolicy: string;
    onSelectionChange: (policy: string) => void;
    isDisabled?: boolean;
    trainingDevice: SchemaDeviceInfo | null;
}

export const PolicySelection = ({
    selectedPolicy,
    onSelectionChange,
    isDisabled,
    trainingDevice,
}: PolicySelectionProps) => {
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
