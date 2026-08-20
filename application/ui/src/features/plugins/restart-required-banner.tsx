import { Button, Flex, Text, View } from '@geti-ui/ui';

import { useRestartState } from './restart-state';
import { useRestartServerMutation } from './plugins.hooks';

const statusText: Record<string, string> = {
    idle: 'Restart the server to activate the plugin changes.',
    requesting: 'Requesting server restart…',
    waiting_for_down: 'Waiting for server to go down…',
    waiting_for_up: 'Waiting for server startup…',
    failed: 'Could not confirm restart from health checks. You can retry.',
};

export const RestartRequiredBanner = ({
    compact = false,
    inFooter = false,
}: {
    compact?: boolean;
    inFooter?: boolean;
}) => {
    const restartMutation = useRestartServerMutation();
    const isRestarting = restartMutation.isPending;
    const { openRestartPrompt } = useRestartState();

    const restart = async () => {
        openRestartPrompt();
    };

    const content = (
        <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
            <Text>{statusText[restartMutation.restartStatus]}</Text>
            <Button
                variant='primary'
                isDisabled={isRestarting}
                onPress={restart}
                UNSAFE_style={
                    compact
                        ? {
                              minHeight: '24px',
                              paddingInline: 'var(--spectrum-global-dimension-size-100)',
                          }
                        : undefined
                }
            >
                {isRestarting ? 'Restarting…' : compact ? 'Restart' : 'Restart server'}
            </Button>
        </Flex>
    );

    return (
        <>
            {inFooter ? (
                content
            ) : (
                <View
                    padding='size-200'
                    borderColor='yellow-400'
                    borderWidth='thin'
                    borderRadius='regular'
                    backgroundColor='yellow-100'
                >
                    {content}
                </View>
            )}
        </>
    );
};
