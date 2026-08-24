import { Divider, Flex, Text, View } from '@geti-ui/ui';

import type { components } from '../../../api/openapi-spec';

type IOFeature = components['schemas']['IOFeature'];

const formatShape = (shape: IOFeature['shape']) => {
    if (shape === null || shape === undefined) {
        return '-';
    }

    return shape.length === 0 ? 'scalar' : `[${shape.join(', ')}]`;
};

const getHeadingColor = (feature: IOFeature) => {
    if (feature.ftype === 'STATE') {
        return 'var(--coral)';
    }

    if (feature.ftype === 'VISUAL') {
        return 'var(--moss-tint-1)';
    }

    if (feature.ftype === 'LANGUAGE') {
        return 'var(--brand-daisy)';
    }

    if (feature.ftype === 'ACTION') {
        return 'var(--energy-blue)';
    }

    return undefined;
};

export const FeatureRow = ({ feature }: { feature: IOFeature }) => {
    const color = getHeadingColor(feature);

    return (
        <View
            backgroundColor={'gray-100'}
            padding='size-100'
            borderRadius={'regular'}
            borderWidth={'thin'}
            borderColor='gray-300'
        >
            <Flex direction='column' gap='size-10'>
                <Text UNSAFE_style={{ fontWeight: 'bold', color }}>{feature.ftype}</Text>
                <Text UNSAFE_style={{ fontWeight: 'bold' }}>{feature.name}</Text>
                <Divider orientation='horizontal' size='S' marginY='size-50' />
                <View marginTop='size-50' UNSAFE_style={{ fontFamily: 'monospace' }}>
                    <Text marginEnd='size-100'>{feature.dtype}</Text>
                    <Text>{formatShape(feature.shape)}</Text>
                </View>
            </Flex>
        </View>
    );
};
