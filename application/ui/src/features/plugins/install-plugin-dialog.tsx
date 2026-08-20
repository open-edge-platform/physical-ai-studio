import { ActionButton, Button, Content, Dialog, Divider, Flex, Heading, Icon, Text, View } from '@geti-ui/ui';
import { Close } from '@geti-ui/ui/icons';

import { SchemaPluginResponse } from '../../api/openapi-spec';
import { usePluginActions } from './plugins.hooks';

export const InstallPluginDialog = ({ plugin, close }: { plugin: SchemaPluginResponse; close: () => void }) => {
    const { isBusy, busyId, install } = usePluginActions();
    const isInstalling = busyId === plugin.id && isBusy;

    return (
        <Dialog size='M' width={'100%'}>
            <Heading>
                <Flex width='100%' justifyContent={'space-between'} alignItems='center'>
                    <span>{plugin.name}</span>
                    <ActionButton isQuiet aria-label='Close' onPress={close}>
                        <Icon>
                            <Close />
                        </Icon>
                    </ActionButton>
                </Flex>
            </Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>{plugin.description}</Text>
                    <View>
                        <Heading level={4}>Robots added</Heading>
                        <Flex direction='column' gap='size-50'>
                            {plugin.robots.map((robot) => (
                                <Text key={robot.type}>{robot.display_name}</Text>
                            ))}
                        </Flex>
                    </View>
                    <Button variant='primary' isDisabled={isBusy} onPress={() => install(plugin.id)}>
                        {isInstalling ? 'Installing…' : 'Install'}
                    </Button>
                </Flex>
            </Content>
        </Dialog>
    );
};
