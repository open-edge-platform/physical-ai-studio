import { Flex, Heading, Text, View } from '@geti-ui/ui';

export const PluginsDisabledView = () => {
    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Flex direction='column' gap='size-150'>
                <Heading level={1}>Plugins</Heading>
                <Text>Plugin management is disabled for this instance.</Text>
            </Flex>
        </View>
    );
};
