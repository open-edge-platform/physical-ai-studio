import { Divider, Flex, Heading, Radio, Text, View } from '@geti-ui/ui';
import { Label } from 'react-aria-components';

import { INFERENCE_BACKENDS } from '../inference-backends';

interface BackendOptionProps {
    id: string;
    isSelected?: boolean;
    isDisabled?: boolean;
}

export const BackendOption = ({ id, isSelected = false, isDisabled = false }: BackendOptionProps) => {
    const backend = INFERENCE_BACKENDS[id];

    return (
        <Label htmlFor={id}>
            <View
                borderColor={isSelected ? 'blue-400' : 'gray-50'}
                borderWidth='thick'
                borderRadius='small'
                padding='size-100'
                backgroundColor={isSelected ? 'gray-50' : 'gray-100'}
                UNSAFE_style={isDisabled ? { opacity: 0.7 } : undefined}
            >
                <Flex direction='row' gap='size-100' alignItems='center'>
                    <View padding='size-200'>
                        <Radio id={id} value={id} marginEnd={0} isDisabled={isDisabled} />
                    </View>

                    <Divider orientation='vertical' size='S' />
                    <Flex height='100%' alignItems='center' justifyContent={'center'} width='size-1000'>
                        <View padding='size-100'>
                            <backend.logo height={'50px'} />
                        </View>
                    </Flex>
                    <View>
                        <Heading level={4}>{backend.label}</Heading>
                        <Text
                            UNSAFE_style={{
                                fontSize: '12px',
                                color: 'var(--spectrum-global-color-gray-700)',
                            }}
                        >
                            {backend.description}
                        </Text>
                    </View>
                </Flex>
            </View>
        </Label>
    );
};
