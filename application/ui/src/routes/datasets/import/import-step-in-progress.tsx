import { Content, Flex, Loading, Text, View } from '@geti-ui/ui';

interface ImportStepInProgressProps {
    statusMessage: string;
}

export const ImportStepInProgress = ({ statusMessage }: ImportStepInProgressProps) => {
    return (
        <>
            <Content>
                <View padding='size-800'>
                    <Flex direction='column' gap='size-400'>
                        <Loading size='M' mode='inline' />
                        <Text
                            UNSAFE_style={{
                                color: 'var(--spectrum-global-color-gray-500)',
                                fontSize: 'var(--spectrum-global-dimension-font-size-100)',
                                textAlign: 'center',
                            }}
                        >
                            {statusMessage}
                        </Text>
                    </Flex>
                </View>
            </Content>
        </>
    );
};
