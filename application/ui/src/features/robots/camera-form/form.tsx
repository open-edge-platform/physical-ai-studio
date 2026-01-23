import { ReactNode } from 'react';

import { Button, Divider, Flex, Heading, Icon, View } from '@geti/ui';
import { ChevronLeft } from '@geti/ui/icons';

import { RadioDisclosure } from '../../../components/radio-disclosure-group/radio-disclosure-group';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { CameraIcon } from './camera-icon';
import { useAllAvailableCameras } from './components/use-camera-data';
import { BaslerFormFields } from './drivers/basler';
import { GenicamFormFields } from './drivers/genicam';
import { IpCamFormFields } from './drivers/ipcam';
import { RealsenseFormFields } from './drivers/realsense';
import { USBCameraFormFields } from './drivers/usb-camera';
import { CameraDriver, useCameraForm, useSetCameraForm } from './provider';
import { SubmitNewCameraButton } from './submit-new-camera-button';
import { UpdateCameraButton } from './update-camera-button';

const Header = ({ heading }: { heading: ReactNode }) => {
    const { project_id } = useProjectId();

    return (
        <Flex alignItems={'center'} gap='size-200'>
            <Button
                href={paths.project.cameras.index({ project_id })}
                variant='secondary'
                UNSAFE_style={{ border: 'none' }}
            >
                <Icon>
                    <ChevronLeft color='white' fill='white' />
                </Icon>
            </Button>

            <Heading>{heading}</Heading>
        </Flex>
    );
};

const CameraFormFields = () => {
    const { state } = useCameraForm();
    const activeDriver = state.activeDriver;
    const { setActiveDriver } = useSetCameraForm();

    const { data: availableHardware } = useAllAvailableCameras();

    const hasDriver = (driver: CameraDriver) => {
        return availableHardware.some(
            (camera) => camera.driver === driver || (driver === 'usb_camera' && camera.driver === 'webcam')
        );
    };

    const items = [
        {
            label: 'USB Camera',
            value: 'usb_camera' as const,
            icon: <CameraIcon type='usb_camera' width={'24px'} />,
            content: <USBCameraFormFields />,
            visible: hasDriver('usb_camera'),
        },
        {
            label: 'Realsense',
            value: 'realsense' as const,
            icon: <CameraIcon type='realsense' width={'24px'} />,
            content: <RealsenseFormFields />,
            visible: hasDriver('realsense'),
        },
        {
            label: 'Basler',
            value: 'basler' as const,
            icon: <CameraIcon type='basler' width={'24px'} />,
            content: <BaslerFormFields />,
            visible: hasDriver('basler'),
        },
        {
            label: 'Genicam',
            value: 'genicam' as const,
            icon: <CameraIcon type='genicam' width={'24px'} />,
            content: <GenicamFormFields />,
            visible: hasDriver('genicam'),
        },
        {
            label: 'IP Camera',
            value: 'ipcam' as const,
            icon: <CameraIcon type='ipcam' width={'24px'} />,
            content: <IpCamFormFields />,
            visible: true, // Always allow IP cameras
        },
    ].filter((item) => item.visible);

    return (
        <RadioDisclosure
            value={activeDriver}
            setValue={(value) => setActiveDriver(value as CameraDriver)}
            items={items}
        />
    );
};

const EditCameraFormFields = () => {
    const { state } = useCameraForm();

    switch (state.activeDriver) {
        case 'usb_camera':
            return <USBCameraFormFields />;
        case 'ipcam':
            return <IpCamFormFields />;
        case 'realsense':
            return <RealsenseFormFields />;
        case 'basler':
            return <BaslerFormFields />;
        case 'genicam':
            return <GenicamFormFields />;
    }
};

export const CameraForm = ({ isEdit = false }) => {
    return (
        <Flex direction='column' gap='size-200'>
            <Header heading={isEdit === false ? 'Add new camera' : 'Update camera'} />

            <Divider orientation='horizontal' size='S' />

            <Flex direction='column' gap='size-200'>
                {isEdit === false ? <CameraFormFields /> : <EditCameraFormFields />}

                <Divider orientation='horizontal' size='S' />

                <View>{isEdit === false ? <SubmitNewCameraButton /> : <UpdateCameraButton />}</View>
            </Flex>
        </Flex>
    );
};
