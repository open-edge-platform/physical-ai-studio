import { Content, Flex, Heading, IllustratedMessage, Text, View } from '@geti/ui';

import { CameraThumbnail } from '../camera-thumbnail';
import { RobotViewer } from '../controller/robot-viewer';
import { ReactComponent as RobotIllustration } from './../../../assets/illustrations/INTEL_08_NO-TESTS.svg';
import { useRobotForm } from './provider';

const EmptyPreview = () => {
    return (
        <IllustratedMessage>
            <RobotIllustration />

            <Flex direction='column' gap='size-200'>
                <Content>
                    <Text>
                        Choose the robot you&aposd like to add using the form on the left. After connecting the camera,
                        the preview will appear here.
                    </Text>
                </Content>
                <Heading>Setup your new robot</Heading>
            </Flex>
        </IllustratedMessage>
    );
};

export const Preview = () => {
    const form = useRobotForm();

    return (
        <View
            backgroundColor={'gray-200'}
            height={'100%'}
            padding='size-200'
            UNSAFE_style={{
                borderRadius: 'var(--spectrum-alias-border-radius-regular)',
                borderColor: 'var(--spectrum-global-color-gray-700)',
                borderWidth: '1px',
                borderStyle: 'dashed',
            }}
            position={'relative'}
        >
            {form.type !== null ? <RobotViewer /> : <EmptyPreview />}

            <View position='absolute' padding='size-100' top={'-20px'} left={'-20px'}>
                <Flex direction='column' gap='size-100'>
                    {form.cameras.map((camera, idx) => {
                        return (
                            <CameraThumbnail
                                key={`${idx}-${camera.fingerprint}`}
                                name={camera.name}
                                fingerprint={camera.fingerprint}
                            />
                        );
                    })}
                </Flex>
            </View>
        </View>
    );
};
