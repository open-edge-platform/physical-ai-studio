import { Content, Flex, Heading, IllustratedMessage, Text, View } from '@geti/ui';

import { CameraFeed } from '../../cameras/camera-feed';
import { ReactComponent as RobotIllustration } from './../../../assets/illustrations/INTEL_08_NO-TESTS.svg';
import { isValid, useCameraForm } from './provider';

const EmptyPreview = () => {
    return (
        <IllustratedMessage>
            <RobotIllustration />

            <Flex direction='column' gap='size-200'>
                <Content>
                    <Text>
                        Choose the camera you&apos; like to add using the form on the left. After connecting the camera,
                        the preview will appear here.
                    </Text>
                </Content>
                <Heading>Setup your new camera</Heading>
            </Flex>
        </IllustratedMessage>
    );
};

export const Preview = () => {
    const { getFormData, state } = useCameraForm();
    const cameraForm = getFormData(state.activeDriver);

    const camera = {
        // Ingore the camera names as this is not important for connection
        hardware_name: cameraForm.hardware_name ?? null,
        // Our server reuses validation requirements that says a camera must have a name
        name: cameraForm.name ?? '_',
        ...cameraForm,
    };

    // Make sure we completely refresh the camera preview when changing camera or resolution
    // eslint-disable-next-line max-len
    const key = `${camera.driver}-${camera.fingerprint}-${camera.payload?.fps}-${camera.payload?.height}-${camera.payload?.width}`;

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
            {isValid(camera) ? <CameraFeed key={key} camera={camera} /> : <EmptyPreview />}
        </View>
    );
};
