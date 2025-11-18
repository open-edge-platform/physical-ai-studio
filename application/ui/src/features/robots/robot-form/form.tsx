import { Suspense } from 'react';

import {
    ActionButton,
    Button,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Form,
    Heading,
    Icon,
    Item,
    Loading,
    Picker,
    Text,
    TextField,
    View,
} from '@geti/ui';
import { Adjustments, ChevronLeft, Close, Refresh } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { SchemaRobotCamera } from '../../../api/openapi-spec';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useRobotForm, useSetRobotForm } from './provider';
import { SubmitNewRobotButton } from './submit-new-robot-button';

import classes from './form.module.scss';

const INITIAL_CAMERA_CONFIGURATION = {
    name: '',
    fingerprint: '',
    resolution_fps: 30,
    resolution_width: 480,
    resolution_height: 360,
};

const RobotType = () => {
    const setRobotForm = useSetRobotForm();
    const robotForm = useRobotForm();

    return (
        <Picker
            isRequired
            label='Robot type'
            width='100%'
            selectedKey={robotForm.type}
            onSelectionChange={(selected) => {
                setRobotForm((oldForm) => ({
                    ...oldForm,
                    type: selected === 'SO101_Follower' ? 'SO101_Follower' : 'SO101_Leader',
                }));
            }}
        >
            <Item key={'SO101_Follower'}>SO101 Follower</Item>
            <Item key={'SO101_Leader'}>SO101 Leader</Item>
        </Picker>
    );
};

const CameraResolutionDialog = ({
    camera,
    updateCamera,
}: {
    camera: SchemaRobotCamera;
    updateCamera: (camera: Partial<SchemaRobotCamera>) => void;
}) => {
    // TODO: based on the selected available camera, get the camera profile
    // then restrict the resolutions based on its profile
    // const availableCamerasQuery = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const SUPPORTED_FPS = [24, 25, 30, 60, 120];
    const SUPPORTED_RESOLUTION = [
        { key: '360p', width: 480, height: 360 },
        { key: '480p', width: 640, height: 480 },
        { key: '720p', width: 1280, height: 720 },
        { key: '1080p', width: 1920, height: 1080 },
        { key: '2160p', width: 3160, height: 2160 },
    ];

    const selectedResolutionKey = SUPPORTED_RESOLUTION.find(
        ({ width, height }) => width === camera.resolution_width && height === camera.resolution_height
    )?.key;

    return (
        <Dialog>
            <Heading>Configure camera profile</Heading>
            <Divider />
            <Content>
                <Flex gap='size-200' direction='column'>
                    <Picker
                        label='Resolution'
                        width='100%'
                        selectedKey={selectedResolutionKey}
                        onSelectionChange={(resolution) => {
                            const selectedResolution = SUPPORTED_RESOLUTION.find(({ key }) => key === resolution);
                            if (selectedResolution === undefined) {
                                return;
                            }

                            updateCamera({
                                resolution_width: selectedResolution.width,
                                resolution_height: selectedResolution.height,
                            });
                        }}
                    >
                        {SUPPORTED_RESOLUTION.map(({ key, width, height }) => {
                            return <Item key={key}>{`${width} x ${height}`}</Item>;
                        })}
                    </Picker>

                    <Picker
                        label='Frames per second (FPS)'
                        width='100%'
                        selectedKey={String(camera.resolution_fps)}
                        onSelectionChange={(fps) => {
                            if (fps === null) {
                                return;
                            }

                            updateCamera({ resolution_fps: Number(fps) });
                        }}
                    >
                        {SUPPORTED_FPS.map((fps) => (
                            <Item key={fps}>{`${fps}`}</Item>
                        ))}
                    </Picker>
                </Flex>
            </Content>
        </Dialog>
    );
};

const Camera = ({ idx, camera, onRemove }: { onRemove: () => void; camera: SchemaRobotCamera; idx: number }) => {
    const availableCamerasQuery = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const setRobotForm = useSetRobotForm();

    const updateCamera = (newCamera: Partial<SchemaRobotCamera>) => {
        setRobotForm((oldForm) => {
            const cameras = oldForm.cameras.map((oldCamera, oldIdx) => {
                return idx === oldIdx ? { ...oldCamera, ...newCamera } : oldCamera;
            });

            return { ...oldForm, cameras };
        });
    };

    return (
        <Flex gap='size-100' alignItems='end'>
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
                    const selected = availableCamerasQuery.data.find(
                        ({ port_or_device_id }) => port_or_device_id === key
                    );

                    if (!selected) {
                        return;
                    }
                    updateCamera({ fingerprint: selected.port_or_device_id });
                }}
            >
                {availableCamerasQuery.data.map((availableCamera) => {
                    return (
                        <Item textValue={availableCamera.port_or_device_id} key={availableCamera.port_or_device_id}>
                            {/* TODO: use an Icon here for visualizing the driver? */}
                            <Text>{availableCamera.name}</Text>
                            <Text slot={'description'}>
                                {availableCamera.port_or_device_id} ({availableCamera.driver})
                            </Text>
                        </Item>
                    );
                })}
            </Picker>
            <DialogTrigger type='popover'>
                <ActionButton UNSAFE_className={classes.actionButton}>
                    <Icon>
                        <Adjustments />
                    </Icon>
                </ActionButton>
                <CameraResolutionDialog camera={camera} updateCamera={updateCamera} />
            </DialogTrigger>
            <ActionButton onPress={onRemove} UNSAFE_className={classes.actionButton}>
                <Icon>
                    <Close />
                </Icon>
            </ActionButton>
        </Flex>
    );
};

const RefreshRobotsButton = () => {
    const { refetch, isFetching } = $api.useSuspenseQuery('get', '/api/hardware/robots');

    return (
        <ActionButton
            isDisabled={isFetching}
            UNSAFE_className={classes.actionButton}
            onPress={() => {
                refetch();
            }}
        >
            <Icon>
                <Refresh />
            </Icon>
        </ActionButton>
    );
};

const IdentifyRobot = () => {
    const { data: robots } = $api.useSuspenseQuery('get', '/api/hardware/robots');

    const robotForm = useRobotForm();
    const identifyMutation = $api.useMutation('post', '/api/hardware/identify');

    const isDisabled = identifyMutation.isPending || robotForm.serial_id === null;
    return (
        <ActionButton
            isDisabled={isDisabled}
            UNSAFE_className={classes.actionButton}
            onPress={() => {
                const body = robots.find((m) => m.serial_id === robotForm.serial_id);

                if (isDisabled || body === undefined) {
                    return;
                }

                identifyMutation.mutate({ body });
            }}
        >
            Identify
        </ActionButton>
    );
};

const CameraFormFields = () => {
    const robotForm = useRobotForm();
    const setRobotForm = useSetRobotForm();
    const addCamera = () => {
        setRobotForm((oldForm) => ({
            ...oldForm,
            cameras: [...oldForm.cameras, INITIAL_CAMERA_CONFIGURATION],
        }));
    };
    const removeCamera = (idxToRemove: number) => {
        setRobotForm((oldForm) => ({
            ...oldForm,
            cameras: oldForm.cameras.filter((_, idx) => idx !== idxToRemove),
        }));
    };

    return (
        <Suspense fallback={<Loading mode='inline' />}>
            {robotForm.cameras.map((camera, idx) => {
                return <Camera key={idx} idx={idx} onRemove={() => removeCamera(idx)} camera={camera} />;
            })}

            <Button variant='primary' onPress={addCamera}>
                Add camera
            </Button>
        </Suspense>
    );
};

export const RobotForm = ({ heading = 'Add new robot', submitButton = <SubmitNewRobotButton /> }) => {
    const { project_id } = useProjectId();

    const robotsQuery = $api.useSuspenseQuery('get', '/api/hardware/robots');

    const robotForm = useRobotForm();
    const setRobotForm = useSetRobotForm();

    return (
        <Flex direction='column' gap='size-200'>
            <Flex alignItems={'center'} gap='size-200'>
                <Button
                    href={paths.project.robots.index({ project_id })}
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
                    <Flex direction='column' gap='size-200' width='100%'>
                        <TextField
                            isRequired
                            label='Robot name'
                            width='100%'
                            onChange={(name) => {
                                setRobotForm((oldForm) => ({ ...oldForm, name }));
                            }}
                            value={robotForm.name}
                        />

                        {/* Put robot type first as we can use it to visualize the robot
                          and determine how to connect with it */}
                        <RobotType />

                        <Flex gap='size-100' justifyContent={'space-between'} alignItems={'end'}>
                            <Picker
                                label='Select robot'
                                isRequired
                                width='100%'
                                selectedKey={robotForm.serial_id}
                                onSelectionChange={(serialId) => {
                                    setRobotForm((oldForm) => ({ ...oldForm, serial_id: String(serialId) }));
                                }}
                            >
                                {robotsQuery.data.map((robot) => {
                                    return (
                                        <Item key={robot.serial_id} textValue={robot.serial_id}>
                                            <Text>{robot.serial_id}</Text>
                                            <Text slot='description'>{robot.port}</Text>
                                        </Item>
                                    );
                                })}
                            </Picker>
                            <Flex gap='size-100'>
                                <RefreshRobotsButton />
                                <IdentifyRobot />
                            </Flex>
                        </Flex>
                    </Flex>
                    <Divider orientation='horizontal' size='S' />
                    <CameraFormFields />
                    <Divider orientation='horizontal' size='S' />
                    <View>{submitButton}</View>
                </Flex>
            </Form>
        </Flex>
    );
};
