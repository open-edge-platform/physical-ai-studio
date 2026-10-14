import { Card, Divider, Flex, Text, View } from '@geti-ui/ui';

import { SchemaAvailablePolicy, SchemaDeviceInfo } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { formatBytes } from './policies';

import classes from './train-model-dialog.module.css';

interface PolicySelectionProps {
    selectedPolicy: string;
    availablePolicies: SchemaAvailablePolicy[];
    isLoadingPolicies: boolean;
    policiesUnavailable?: boolean;
    onSelectionChange: (policy: string) => void;
    isDisabled?: boolean;
    trainingDevice: SchemaDeviceInfo | null;
}

export const PolicySelection = ({
    selectedPolicy,
    availablePolicies,
    isLoadingPolicies,
    policiesUnavailable,
    onSelectionChange,
    isDisabled,
    trainingDevice,
}: PolicySelectionProps) => {
    const availableVram = trainingDevice?.memory ?? 0;

    const selectedModel = availablePolicies.find((model) => model.id === selectedPolicy) ?? null;
    const hasInsufficientVram =
        selectedModel?.min_vram != null &&
        availableVram > 0 &&
        selectedModel.min_vram > availableVram &&
        formatBytes(selectedModel.min_vram) !== formatBytes(availableVram);

    return (
        <Flex direction='column' gap='size-100'>
            <Text UNSAFE_style={{ fontSize: 12 }}>Policy</Text>
            <div className={classes.policyGrid}>
                {availablePolicies.map((model) => {
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
                                        {model.min_vram != null && <Text>&ge; {formatBytes(model.min_vram)} VRAM</Text>}
                                    </Flex>
                                </Flex>
                                <Divider size='S' />
                                <Text UNSAFE_style={{ fontSize: 12 }}>{model.description}</Text>
                            </Flex>
                        </Card>
                    );
                })}
            </div>

            {policiesUnavailable && (
                <InlineAlert variant='warning'>
                    Cannot verify policies on this training target. Check its connection.
                </InlineAlert>
            )}
            {!isLoadingPolicies && !policiesUnavailable && availablePolicies.length === 0 && (
                <InlineAlert variant='warning'>No training policies are available. Check the backend logs.</InlineAlert>
            )}
            {!isLoadingPolicies && availablePolicies.length > 0 && !selectedModel && (
                <InlineAlert variant='warning'>This policy is unavailable. Choose another policy.</InlineAlert>
            )}
            {hasInsufficientVram && selectedModel?.min_vram != null && (
                <View marginTop='size-100'>
                    <InlineAlert variant='warning'>
                        {selectedModel.name} requires at least {formatBytes(selectedModel.min_vram)} VRAM, but the
                        device reports less than that. Training may fail or be very slow.
                    </InlineAlert>
                </View>
            )}
        </Flex>
    );
};
