import { Content, ContextualHelp, Flex, Heading, Item, Picker, ProgressCircle, Text } from '@geti-ui/ui';

import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { MODELS } from './policies';
import { FeatureMapping } from './use-feature-mapping';

/** Picker key standing in for "no camera in this slot". */
const EMPTY_SLOT_KEY = '__empty__';

interface FeatureMappingStepProps {
    policy: string;
    mapping: FeatureMapping;
}

/**
 * Maps the dataset's cameras onto the fixed camera slots of the selected policy.
 *
 * A policy pretrained on a fixed camera order reads its first slot as the scene
 * view, its second as the wrist view, and so on. A dataset just has cameras, so
 * this is where the two are lined up. Policies without such an order skip this
 * step altogether (see `getWizardSteps`), so `mapping.slots` is never empty here.
 */
export const FeatureMappingStep = ({ policy, mapping }: FeatureMappingStepProps) => {
    const policyName = MODELS.find((model) => model.id === policy)?.name ?? policy;
    const { slots, cameras, cameraKeyBySlotId, assignCamera, isLoading, isDatasetEmpty, error } = mapping;

    if (isLoading) {
        return (
            <Flex alignItems='center' justifyContent='center' height='100%' width='100%'>
                <ProgressCircle aria-label='Loading dataset cameras' isIndeterminate />
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap='size-200' width='100%'>
            <Text>
                {policyName} is pretrained on a fixed camera order. Pick the dataset camera that fills each slot; a slot
                left empty is trained on a masked empty image.
            </Text>

            {isDatasetEmpty && (
                <InlineAlert variant='warning'>
                    This dataset has no episodes yet, so its cameras are unknown. Record an episode first.
                </InlineAlert>
            )}

            {slots.map((slot) => {
                const selectedCameraKey = cameraKeyBySlotId[slot.id] ?? null;

                return (
                    <Flex key={slot.id} direction='row' gap='size-100' alignItems='end' width='100%'>
                        <Picker
                            width='100%'
                            label={slot.name}
                            isRequired={slot.isRequired}
                            necessityIndicator='label'
                            placeholder='Empty'
                            isDisabled={cameras.length === 0}
                            selectedKey={selectedCameraKey ?? EMPTY_SLOT_KEY}
                            onSelectionChange={(key) =>
                                assignCamera(slot.id, key === EMPTY_SLOT_KEY ? null : String(key))
                            }
                        >
                            {[{ key: EMPTY_SLOT_KEY, name: 'Empty' }, ...cameras].map((option) => (
                                <Item key={option.key}>{option.name}</Item>
                            ))}
                        </Picker>
                        <ContextualHelp variant='info'>
                            <Heading>{slot.name}</Heading>
                            <Content>
                                <Text>{slot.description}</Text>
                            </Content>
                        </ContextualHelp>
                    </Flex>
                );
            })}

            {error !== null && <InlineAlert variant='warning'>{error}</InlineAlert>}
        </Flex>
    );
};
