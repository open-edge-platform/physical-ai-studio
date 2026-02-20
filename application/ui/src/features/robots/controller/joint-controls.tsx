import { useCallback, useEffect, useState } from 'react';

import { ActionButton, Flex, Grid, Heading, minmax, repeat, Slider, Switch, View } from '@geti/ui';
import { ChevronDownSmallLight } from '@geti/ui/icons';
import useWebSocket from 'react-use-websocket';
import { degToRad, radToDeg } from 'three/src/math/MathUtils.js';

import { useRobotModels } from '../robot-models-context';
import { useRobot, useRobotId } from '../use-robot';

function removeSuffix(str: string, suffix: string): string {
    return str.endsWith(suffix) ? str.slice(0, -suffix.length) : str;
}

const Joint = ({
    name,
    value,
    minValue,
    maxValue,
    decreaseKey,
    increaseKey,
    isDisabled,
    onChange,
}: {
    name: string;
    value: number;
    minValue: number;
    maxValue: number;
    decreaseKey: string;
    increaseKey: string;
    isDisabled: boolean;
    onChange: (value: number) => void;
}) => {
    const [state, setState] = useState(value);
    useEffect(() => {
        if (isDisabled) {
            setState(value);
        }
    }, [value, isDisabled]);

    return (
        <li>
            <View
                backgroundColor={'gray-50'}
                padding='size-115'
                UNSAFE_style={{
                    //border: '1px solid var(--spectrum-global-color-gray-200)',
                    borderRadius: '4px',
                }}
            >
                <Grid areas={['name value', 'slider slider']} gap='size-100'>
                    <div style={{ gridArea: 'name' }}>
                        <span>{name}</span>
                    </div>
                    <div style={{ gridArea: 'value', display: 'flex', justifyContent: 'end' }}>
                        <span style={{ color: 'var(--energy-blue-light)' }}>{value.toFixed(2)}&deg;</span>
                    </div>
                    <Flex gridArea='slider' gap='size-200'>
                        <View
                            backgroundColor={'gray-100'}
                            paddingY='size-50'
                            paddingX='size-150'
                            UNSAFE_style={{
                                borderRadius: '4px',
                            }}
                        >
                            <kbd>{decreaseKey}</kbd>
                        </View>
                        <Slider
                            aria-label={name}
                            value={state}
                            defaultValue={value}
                            minValue={minValue}
                            maxValue={maxValue}
                            flexGrow={1}
                            isDisabled={isDisabled}
                            onChangeEnd={isDisabled ? undefined : onChange}
                            onChange={setState}
                        />
                        <View
                            backgroundColor={'gray-100'}
                            paddingY='size-50'
                            paddingX='size-150'
                            UNSAFE_style={{
                                borderRadius: '4px',
                            }}
                        >
                            <kbd>{increaseKey}</kbd>
                        </View>
                    </Flex>
                </Grid>
            </View>
        </li>
    );
};

type JointsState = Array<{
    name: string;
    value: number;
    rangeMin: number;
    rangeMax: number;
    decreaseKey: string;
    increaseKey: string;
}>;
const useJointState = () => {
    const [isControlled, setIsControlled] = useState(false);
    const [joints, setJoints] = useState<JointsState>([]);
    const { models } = useRobotModels();

    // WebSocket message handler
    const handleMessage = useCallback(
        (event: WebSocketEventMap['message']) => {
            try {
                const payload = JSON.parse(event.data);

                if (payload['event'] === 'state_was_updated') {
                    const newJoints = payload['state'];

                    Object.keys(newJoints).forEach((joint) => {
                        const name = removeSuffix(joint, '.pos');

                        models.forEach((model) => {
                            model.setJointValue(name, degToRad(newJoints[joint]));
                        });
                    });

                    const placeholderJoints = [
                        { name: 'J1', value: 70, rangeMin: -360, rangeMax: 360, decreaseKey: 'q', increaseKey: '1' },
                        { name: 'J2', value: 20, rangeMin: -360, rangeMax: 360, decreaseKey: '2', increaseKey: '2' },
                        { name: 'J3', value: 80, rangeMin: -360, rangeMax: 360, decreaseKey: 'e', increaseKey: '3' },
                        { name: 'J4', value: 60, rangeMin: -360, rangeMax: 360, decreaseKey: 'r', increaseKey: '4' },
                        { name: 'J5', value: 10, rangeMin: -360, rangeMax: 360, decreaseKey: 't', increaseKey: '5' },
                        { name: 'J6', value: 84, rangeMin: -360, rangeMax: 360, decreaseKey: 'y', increaseKey: '6' },
                    ];

                    const modelJoints = Object.values(models.at(0)?.joints) ?? [];

                    const jointState = Object.keys(newJoints).map((joint_name, idx) => {
                        const joint = modelJoints.find(({ urdfName }) => urdfName === joint_name);

                        const rangeMax = joint === undefined ? 180 : radToDeg(joint.limit.upper);
                        const rangeMin = joint === undefined ? -180 : radToDeg(joint.limit.lower);

                        return {
                            ...placeholderJoints[idx],
                            name: joint_name,
                            value: Number(newJoints[joint_name]),
                            rangeMax,
                            rangeMin,
                        };
                    });

                    setJoints(jointState);
                    setIsControlled(payload['is_controlled']);
                }
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        },
        [models]
    );

    const { project_id, robot_id } = useRobotId();
    const robot = useRobot();
    console.log({ robot });
    //useWebSocket('ws://localhost:8008/api/cameras/ws', {
    const socket = useWebSocket(`/api/projects/${project_id}/robots/${robot_id}/ws`, {
        //const socket = useWebSocket(`ws://localhost:8008/api/robot/ws`, {
        queryParams: {
            driver: 'feetech',
            //serial_id: robot.payload.serial_id,
            // normalize: true,
            // TODO ...
            // fps: 120,
            fps: 30,
        },
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });

    const setJoint = (name: string, value: number) => {
        socket.sendJsonMessage({
            command: 'set_joints_state',
            joints: {
                [name]: value,
            },
        });
    };

    return [joints, isControlled, setJoint, socket] as const;
};

const Joints = ({
    joints,
    setJoint,
    isDisabled,
}: {
    joints: JointsState;
    setJoint: (name: string, value: number) => void;
    isDisabled: boolean;
}) => {
    return (
        <ul>
            <Grid gap='size-50' columns={repeat('auto-fit', minmax('size-4600', '1fr'))}>
                {joints.map((joint) => {
                    return (
                        <Joint
                            isDisabled={isDisabled}
                            key={joint.name}
                            name={joint.name}
                            value={joint.value}
                            minValue={joint.rangeMin}
                            maxValue={joint.rangeMax}
                            decreaseKey={joint.decreaseKey}
                            increaseKey={joint.increaseKey}
                            onChange={(value) => {
                                setJoint(joint.name, value);
                            }}
                        />
                    );
                })}
            </Grid>
        </ul>
    );
};

const CompoundMovements = () => {
    return null;
    return (
        <>
            <Heading level={4}>Compound movements</Heading>
            <ul>
                <Flex gap='size-50'>
                    <li>
                        <View
                            backgroundColor={'gray-50'}
                            padding='size-115'
                            UNSAFE_style={{
                                //border: '1px solid var(--spectrum-global-color-gray-200)',
                                borderRadius: '4px',
                            }}
                        >
                            <Flex gap='size-100' alignItems={'center'}>
                                <span>Jaw down & up</span>
                                <View
                                    backgroundColor={'gray-100'}
                                    paddingY='size-50'
                                    paddingX='size-150'
                                    UNSAFE_style={{
                                        borderRadius: '4px',
                                    }}
                                >
                                    <kbd>i</kbd>
                                </View>
                                <View
                                    backgroundColor={'gray-100'}
                                    paddingY='size-50'
                                    paddingX='size-150'
                                    UNSAFE_style={{
                                        borderRadius: '4px',
                                    }}
                                >
                                    <kbd>8</kbd>
                                </View>
                            </Flex>
                        </View>
                    </li>
                    <li>
                        <View
                            backgroundColor={'gray-50'}
                            padding='size-115'
                            UNSAFE_style={{
                                //border: '1px solid var(--spectrum-global-color-gray-200)',
                                borderRadius: '4px',
                            }}
                        >
                            <Flex gap='size-100' alignItems={'center'}>
                                <span>Jaw backward & forward</span>
                                <View
                                    backgroundColor={'gray-100'}
                                    paddingY='size-50'
                                    paddingX='size-150'
                                    UNSAFE_style={{
                                        borderRadius: '4px',
                                    }}
                                >
                                    <kbd>u</kbd>
                                </View>
                                <View
                                    backgroundColor={'gray-100'}
                                    paddingY='size-50'
                                    paddingX='size-150'
                                    UNSAFE_style={{
                                        borderRadius: '4px',
                                    }}
                                >
                                    <kbd>o</kbd>
                                </View>
                            </Flex>
                        </View>
                    </li>
                </Flex>
            </ul>
        </>
    );
};

export const JointControls = () => {
    const [collapsed, setCollapsed] = useState(false);
    const [joints, isControlled, setJoint, socket] = useJointState();

    return (
        <View
            isHidden
            gridArea='controls'
            backgroundColor={'gray-100'}
            padding='size-100'
            UNSAFE_style={{
                border: '1px solid var(--spectrum-global-color-gray-200)',
                borderRadius: '8px',
            }}
        >
            <Flex direction='column' gap='size-50'>
                <Flex justifyContent={'space-between'}>
                    <ActionButton onPress={() => setCollapsed((c) => !c)}>
                        <Heading level={4} marginX='size-100'>
                            <Flex alignItems='center' gap='size-100'>
                                <ChevronDownSmallLight
                                    fill='white'
                                    style={{
                                        transform: collapsed ? '' : 'rotate(180deg)',
                                        animation: 'transform ease-in-out 0.1s',
                                    }}
                                />
                                Joint state
                            </Flex>
                        </Heading>
                    </ActionButton>

                    <Switch
                        isSelected={isControlled}
                        onChange={(value) => {
                            if (value === true) {
                                socket.sendJsonMessage({ command: 'enable_torque' });
                            } else {
                                socket.sendJsonMessage({ command: 'disable_torque' });
                            }
                        }}
                    >
                        Take control
                    </Switch>
                </Flex>
                {collapsed === false && (
                    <>
                        <Joints joints={joints} isDisabled={isControlled === false} setJoint={setJoint} />
                        <CompoundMovements />
                    </>
                )}
            </Flex>
        </View>
    );
};
