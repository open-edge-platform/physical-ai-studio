import { Content, Flex, Heading, IllustratedMessage, Text, View } from '@geti-ui/ui';

import { RobotViewer } from '../controller/robot-viewer';
import { useRobotCatalogQuery } from '../robot-catalog.hooks';
import { ReactComponent as RobotIllustration } from './../../../assets/illustrations/INTEL_08_NO-TESTS.svg';
import { useRobotForm } from './provider';

const EmptyPreview = () => {
    return (
        <IllustratedMessage>
            <RobotIllustration />

            <Flex direction='column' gap='size-200'>
                <Content>
                    <Text>
                        Choose the robot you&apos; like to add using the form on the left. After connecting the robot,
                        the preview will appear here.
                    </Text>
                </Content>
                <Heading>Setup your new robot</Heading>
            </Flex>
        </IllustratedMessage>
    );
};

const SimulatedPreview = () => (
    <IllustratedMessage>
        <RobotIllustration />
        <Flex direction='column' gap='size-200'>
            <Content>
                <Text>
                    This robot is simulated: its 3D view comes from the simulation. Add it, then start its simulation
                    from the robot list or the robot page.
                </Text>
            </Content>
            <Heading>Simulated robot</Heading>
        </Flex>
    </IllustratedMessage>
);

export const Preview = () => {
    const { activeType } = useRobotForm();
    const { data: catalog } = useRobotCatalogQuery();
    const definition = catalog.find(({ type }) => type === activeType);
    const hasModel = definition !== undefined && Object.keys(definition.package_map ?? {}).length > 0;

    return (
        <View backgroundColor={'gray-200'} height={'100%'}>
            {activeType == null ? (
                <EmptyPreview />
            ) : hasModel || definition?.simulation == null ? (
                <RobotViewer robot={{ type: activeType }} />
            ) : (
                <SimulatedPreview />
            )}
        </View>
    );
};
