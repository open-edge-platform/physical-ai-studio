import { Button, Flex, Text, View } from '@geti-ui/ui';

import { useRestartState } from './restart-state';

export const RestartRequiredBanner = () => {
    const { restartRequired, isRestarting, openRestartPrompt } = useRestartState();

    if (!restartRequired) {
        return null;
    }

    return (
        <View backgroundColor={'yellow-400'} paddingX='size-400'>
            <Flex
                alignItems='center'
                justifyContent='space-between'
                gap='size-100'
                UNSAFE_style={{
                    color: 'black',
                }}
            >
                <Text>
                    {isRestarting ? 'Restarting the server…' : 'Restart the server to activate the plugin changes.'}
                </Text>
                <Button
                    variant='primary'
                    style='fill'
                    isDisabled={isRestarting}
                    onPress={openRestartPrompt}
                    UNSAFE_style={{
                        minHeight: '24px',
                        paddingInline: 'var(--spectrum-global-dimension-size-200)',
                        borderRadius: 0,
                    }}
                >
                    {isRestarting ? 'Restarting…' : 'Restart server'}
                </Button>
            </Flex>
        </View>
    );
};
