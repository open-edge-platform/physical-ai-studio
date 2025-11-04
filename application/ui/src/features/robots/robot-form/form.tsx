import {
    ActionButton,
    Button,
    Divider,
    Flex,
    Form,
    Heading,
    Icon,
    Item,
    Picker,
    Text,
    TextField,
    View,
} from '@geti/ui';
import { ChevronLeft, Close, Refresh } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useRobotForm, useSetRobotForm } from './provider';
import { SubmitNewRobotButton } from './submit-new-robot-button';

import classes from './form.module.scss';

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

type CameraType = {
    name: string;
    fingerprint: string | null;
};

const Camera = ({ idx, camera, onRemove }: { onRemove: () => void; camera: CameraType; idx: number }) => {
    const availableCamerasQuery = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const setRobotForm = useSetRobotForm();

    return (
        <Flex gap='size-100' alignItems='end'>
            <Picker
                label='Camera type'
                width='100%'
                selectedKey={camera.fingerprint}
                onSelectionChange={(key) => {
                    const selected = availableCamerasQuery.data.find(
                        ({ port_or_device_id }) => port_or_device_id === key
                    );

                    if (!selected) {
                        return;
                    }

                    setRobotForm((oldForm) => {
                        const cameras = oldForm.cameras.map((oldCamera, oldIdx) => {
                            return idx === oldIdx
                                ? { ...oldCamera, fingerprint: selected.port_or_device_id }
                                : oldCamera;
                        });

                        return { ...oldForm, cameras };
                    });
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
            <TextField
                isRequired
                label='Camera name'
                width='100%'
                onChange={(name) => {
                    setRobotForm((oldForm) => {
                        const cameras = oldForm.cameras.map((oldCamera, oldIdx) => {
                            return idx === oldIdx ? { ...oldCamera, name } : oldCamera;
                        });

                        return { ...oldForm, cameras };
                    });
                }}
                value={camera.name}
            />
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
            cameras: [...oldForm.cameras, { name: '', fingerprint: '' }],
        }));
    };
    const removeCamera = (idxToRemove: number) => {
        setRobotForm((oldForm) => ({
            ...oldForm,
            cameras: oldForm.cameras.filter((_, idx) => idx !== idxToRemove),
        }));
    };

    return (
        <>
            {robotForm.cameras.map((camera, idx) => {
                return <Camera key={idx} idx={idx} onRemove={() => removeCamera(idx)} camera={camera} />;
            })}

            <Button variant='primary' onPress={addCamera}>
                Add camera
            </Button>
        </>
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
