import { Content, Flex, Heading, IllustratedMessage, Text, View } from '@geti/ui';

import { $api } from '../../../api/client';
import { CameraFeed } from '../../cameras/camera-feed';
import { ReactComponent as RobotIllustration } from './../../../assets/illustrations/INTEL_08_NO-TESTS.svg';
import { useCameraForm } from './provider';

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
    const form = useCameraForm();

    const { data: hardwareCameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const actualCamera = useCameraForm();
    const hardwareCamera = hardwareCameras.find(({ fingerprint }) => {
        return actualCamera.fingerprint === fingerprint;
    });

    const camera = {
        name: actualCamera.name,
        hardware_name: hardwareCamera?.name ?? '',
        driver: hardwareCamera?.driver ?? '',
        fingerprint: hardwareCamera?.fingerprint ?? '',
        fps: actualCamera.resolution_fps,
        width: actualCamera.resolution_width,
        height: actualCamera.resolution_height,
    };

    const isEnabled =
        actualCamera.resolution_fps &&
        actualCamera.resolution_width &&
        actualCamera.resolution_height &&
        actualCamera.fingerprint;

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
            {isEnabled ? (
                <CameraFeed
                    key={`${camera.fingerprint}-${form.resolution_fps}-${form.resolution_height}-${form.resolution_width}`}
                    camera={camera}
                />
            ) : (
                <EmptyPreview />
            )}
        </View>
    );
};
