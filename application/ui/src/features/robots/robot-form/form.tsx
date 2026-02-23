import { useEffect } from 'react';

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
import { ChevronLeft, Refresh } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { SchemaRobot } from '../../../api/openapi-spec';
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
                const newType = selected as typeof robotForm.type;

                const wasSerial = robotForm.type?.toLowerCase().startsWith('so101') ?? false;
                const isSerial = newType?.toLowerCase().startsWith('so101') ?? false;

                setRobotForm((oldForm) => ({
                    ...oldForm,
                    type: newType,
                    ...(wasSerial !== isSerial
                        ? {
                              // Only reset when switching families (SO -> Trossen, etc)
                              serial_number: '',
                              connection_string: '',
                          }
                        : {}),
                }));
            }}
        >
            <Item key={'SO101_Follower'}>SO101 Follower</Item>
            <Item key={'SO101_Leader'}>SO101 Leader</Item>
            <Item key={'Trossen_WidowXAI_Follower'}>Trossen WidowX AI Follower</Item>
            <Item key={'Trossen_WidowXAI_Leader'}>Trossen WidowX AI Leader</Item>
        </Picker>
    );
};

const RefreshRobotsButton = () => {
    const { refetch, isFetching } = $api.useSuspenseQuery('get', '/api/hardware/serial_devices');

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
    const robotForm = useRobotForm();
    const identifyMutation = $api.useMutation('post', '/api/hardware/identify');

    const isDisabled = identifyMutation.isPending || !robotForm.name || !robotForm.type || !robotForm.connection_string;

    const onIdentify = () => {
        if (isDisabled || robotForm.type === null) {
            return;
        }

        const body: SchemaRobot = {
            id: crypto.randomUUID(), // required by schema, not used by backend
            name: robotForm.name,
            type: robotForm.type,
            connection_string: robotForm.connection_string ?? '',
            serial_number: robotForm.serial_number ?? '',
            active_calibration_id: null,
        };

        identifyMutation.mutate({ body });
    };

    return (
        <ActionButton isDisabled={isDisabled} UNSAFE_className={classes.actionButton} onPress={onIdentify}>
            Identify
        </ActionButton>
    );
};

export const RobotForm = ({ heading = 'Add new robot', submitButton = <SubmitNewRobotButton /> }) => {
    const { project_id } = useProjectId();

    const serialDevicesQuery = $api.useSuspenseQuery('get', '/api/hardware/serial_devices');

    const robotForm = useRobotForm();
    const setRobotForm = useSetRobotForm();

    // Since project won't save connection_string for Serial devices;
    // we need to populate this value; so the identify button works.
    useEffect(() => {
        if (!robotForm.serial_number) {
            return;
        }

        const device = serialDevicesQuery.data.find((d) => d.serial_number === robotForm.serial_number);

        if (!device) {
            return;
        }

        if (robotForm.connection_string !== device.connection_string) {
            setRobotForm((oldForm) => ({
                ...oldForm,
                connection_string: device.connection_string,
            }));
        }
    }, [robotForm.serial_number, serialDevicesQuery.data, robotForm.connection_string, setRobotForm]);

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
                            {robotForm.type?.toLowerCase().startsWith('trossen') ? (
                                <>
                                    <TextField
                                        isRequired
                                        label='Robot IP address'
                                        width='100%'
                                        value={robotForm.connection_string ?? ''}
                                        onChange={(connection_string) => {
                                            setRobotForm((oldForm) => ({
                                                ...oldForm,
                                                connection_string,
                                                serial_number: '',
                                            }));
                                        }}
                                        placeholder='192.168.1.2'
                                    />
                                    <Flex gap='size-100'>
                                        <IdentifyRobot />
                                    </Flex>
                                </>
                            ) : null}
                            {robotForm.type?.toLowerCase().startsWith('so101') ? (
                                <>
                                    <Picker
                                        label='Select robot'
                                        isRequired
                                        width='100%'
                                        selectedKey={robotForm.serial_number}
                                        onSelectionChange={(serial_number) => {
                                            const device = serialDevicesQuery.data.find(
                                                (d) => d.serial_number === serial_number
                                            );

                                            setRobotForm((oldForm) => ({
                                                ...oldForm,
                                                serial_number: String(serial_number),
                                                connection_string: device?.connection_string ?? '',
                                            }));
                                        }}
                                    >
                                        {serialDevicesQuery.data.map((serial_device) => {
                                            return (
                                                <Item
                                                    key={serial_device.serial_number}
                                                    textValue={serial_device.serial_number}
                                                >
                                                    <Text>{serial_device.serial_number}</Text>
                                                    <Text slot='description'>{serial_device.connection_string}</Text>
                                                </Item>
                                            );
                                        })}
                                    </Picker>

                                    <Flex gap='size-100'>
                                        <RefreshRobotsButton />
                                        <IdentifyRobot />
                                    </Flex>
                                </>
                            ) : null}
                        </Flex>
                    </Flex>
                    <Divider orientation='horizontal' size='S' />
                    <View>{submitButton}</View>
                </Flex>
            </Form>
        </Flex>
    );
};
