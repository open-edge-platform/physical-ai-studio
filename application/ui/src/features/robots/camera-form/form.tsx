import { Button, Divider, Flex, Form, Heading, Icon, Item, Picker, Text, TextField, View } from '@geti/ui';
import { ChevronLeft } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { CameraFormProps, useCameraForm, useSetCameraForm } from './provider';
import { SubmitNewCameraButton } from './submit-new-camera-button';

const CameraFormFields = () => {
    const camera = useCameraForm();

    const availableCamerasQuery = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const setCameraForm = useSetCameraForm();

    const updateCamera = (newCamera: CameraFormProps) => {
        setCameraForm((oldForm) => {
            return {
                ...oldForm,
                name: newCamera.name ?? oldForm.name,
                fingerprint: newCamera.fingerprint ?? oldForm.fingerprint,
                payload: {
                    fps: newCamera.payload?.fps ?? oldForm.payload?.fps ?? 30,
                    width: newCamera.payload?.width ?? oldForm.payload?.width ?? 640,
                    height: newCamera.payload?.height ?? oldForm.payload?.height ?? 480,
                },
            };
        });
    };

    const supportedFormatsQuery = $api.useQuery(
        'get',
        '/api/cameras/supported_formats/{driver}',
        {
            params: {
                path: { driver: 'webcam' },
                query: { fingerprint: camera.fingerprint ?? '' },
            },
        },
        { enabled: camera.fingerprint !== undefined }
    );

    const supportedResolution = supportedFormatsQuery.data;
    const SUPPORTED_RESOLUTION = supportedResolution ?? [];

    const selectedResolution = SUPPORTED_RESOLUTION.find(
        ({ width, height }) => width === camera.payload?.width && height === camera.payload?.height
    );
    const selectedResolutionKey = `${selectedResolution?.width}_${selectedResolution?.height}`;
    const SUPPORTED_FPS = selectedResolution?.fps ?? [];

    return (
        <Flex gap='size-100' alignItems='end' direction={'column'}>
            <TextField
                isRequired
                label='name'
                width='100%'
                onChange={(name) => {
                    updateCamera({ name });
                }}
                value={camera.name}
            />
            <Picker
                label='Camera'
                width='100%'
                selectedKey={camera.fingerprint}
                onSelectionChange={(key) => {
                    const selected = availableCamerasQuery.data.find(({ fingerprint }) => fingerprint === key);

                    if (!selected) {
                        return;
                    }
                    updateCamera({ fingerprint: selected.fingerprint });
                }}
            >
                {availableCamerasQuery.data.map((availableCamera) => {
                    return (
                        <Item textValue={availableCamera.fingerprint} key={availableCamera.fingerprint}>
                            {/* TODO: use an Icon here for visualizing the driver? */}
                            <Text>{availableCamera.name}</Text>
                            <Text slot={'description'}>
                                {availableCamera.fingerprint} ({availableCamera.driver})
                            </Text>
                        </Item>
                    );
                })}
            </Picker>

            <Picker
                label='Resolution'
                width='100%'
                selectedKey={selectedResolutionKey}
                onSelectionChange={(resolution) => {
                    const newlySelectedResolution = SUPPORTED_RESOLUTION.find(
                        ({ width, height }) => `${width}_${height}` === resolution
                    );
                    if (newlySelectedResolution === undefined) {
                        return;
                    }

                    const newFps =
                        newlySelectedResolution.fps.find((fps) => fps === camera.payload?.fps) ??
                        newlySelectedResolution.fps.at(0) ??
                        1;

                    updateCamera({
                        payload: {
                            width: newlySelectedResolution.width,
                            height: newlySelectedResolution.height,
                            fps: newFps,
                        },
                    });
                }}
            >
                {SUPPORTED_RESOLUTION.map(({ width, height }) => {
                    return <Item key={`${width}_${height}`}>{`${width} x ${height}`}</Item>;
                })}
            </Picker>

            <Picker
                label='Frames per second (FPS)'
                width='100%'
                selectedKey={String(camera.payload?.fps)}
                onSelectionChange={(fps) => {
                    if (fps === null) {
                        return;
                    }

                    updateCamera({ payload: { ...camera.payload!, fps: Number(fps) } });
                }}
            >
                {SUPPORTED_FPS.map((fps) => (
                    <Item key={fps}>{`${fps}`}</Item>
                ))}
            </Picker>
        </Flex>
    );
};

export const CameraForm = ({ heading = 'Add new camera', submitButton = <SubmitNewCameraButton /> }) => {
    const { project_id } = useProjectId();

    return (
        <Flex direction='column' gap='size-200'>
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
            <Divider orientation='horizontal' size='S' />
            <Form>
                <Flex direction='column' gap='size-200'>
                    <CameraFormFields />
                    <Divider orientation='horizontal' size='S' />
                    <View>{submitButton}</View>
                </Flex>
            </Form>
        </Flex>
    );
};
