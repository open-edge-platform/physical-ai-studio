import { useState } from 'react';

import { ActionButton, Button, Flex, Grid, Heading, minmax, repeat, Slider, ToggleButtons, View } from '@geti/ui';
import { ChevronDownSmallLight } from '@geti/ui/icons';
import { useParams } from 'react-router';

import { $api } from '../../../api/client';

const useRobot = () => {
    const params = useParams<{ project_id: string; robot_id: string }>() as {
        project_id: string;
        robot_id: string;
    };

    const robots = $api.useSuspenseQuery('get', '/api/hardware/robots');

    return robots.data.find((robot) => robot.serial_id === params.robot_id);
};

const Joint = ({
    name,
    value,
    minValue,
    maxValue,
    decreaseKey,
    increaseKey,
}: {
    name: string;
    value: number;
    minValue: number;
    maxValue: number;
    decreaseKey: string;
    increaseKey: string;
}) => {
    const robot = useRobot();
    const params = useParams<{ project_id: string; robot_id: string }>() as {
        project_id: string;
        robot_id: string;
    };
    const moveJointMutation = $api.useMutation('put', '/api/hardware/robots/{robot_id}/joints/{joint}');

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
                        <span style={{ color: 'var(--energy-blue-light)' }}>{value}</span>
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
                            defaultValue={value}
                            minValue={minValue}
                            maxValue={maxValue}
                            onChangeEnd={(x) => {
                                console.log('noew', {
                                    name,
                                    value,
                                    x,
                                    minValue,
                                    maxValue,
                                });

                                if (robot === undefined) {
                                    return;
                                }

                                moveJointMutation.mutate({
                                    params: {
                                        path: {
                                            joint: name,
                                        },
                                        query: {
                                            joint_value: x,
                                        },
                                    },
                                    body: {
                                        device_name: robot.device_name,
                                        port: robot.port,
                                        serial_id: robot.serial_id,
                                    },
                                });
                            }}
                            flexGrow={1}
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

const Joints = () => {
    const joints = [
        { name: 'J1', value: 70, rangeMin: 0, rangeMax: 90, decreaseKey: 'q', increaseKey: '1' },
        { name: 'J2', value: 20, rangeMin: 0, rangeMax: 90, decreaseKey: '2', increaseKey: '2' },
        { name: 'J3', value: 80, rangeMin: 0, rangeMax: 90, decreaseKey: 'e', increaseKey: '3' },
        { name: 'J4', value: 60, rangeMin: 0, rangeMax: 90, decreaseKey: 'r', increaseKey: '4' },
        { name: 'J5', value: 10, rangeMin: 0, rangeMax: 90, decreaseKey: 't', increaseKey: '5' },
        { name: 'J6', value: 84, rangeMin: 0, rangeMax: 90, decreaseKey: 'y', increaseKey: '6' },
    ];

    return (
        <ul>
            <Grid gap='size-50' columns={repeat('auto-fit', minmax('size-6000', '1fr'))}>
                {joints.map((joint) => {
                    return (
                        <Joint
                            key={joint.name}
                            name={joint.name}
                            value={joint.value}
                            minValue={joint.rangeMin}
                            maxValue={joint.rangeMax}
                            decreaseKey={joint.decreaseKey}
                            increaseKey={joint.increaseKey}
                        />
                    );
                })}
            </Grid>
        </ul>
    );
};

const CompoundMovements = () => {
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

    return (
        <View
            gridArea='controls'
            backgroundColor={'gray-100'}
            padding='size-100'
            UNSAFE_style={{
                border: '1px solid var(--spectrum-global-color-gray-200)',
                borderRadius: '8px',
            }}
        >
            <Flex direction='column' gap='size-50'>
                <div>
                    <ActionButton onPress={() => setCollapsed((collapsed) => !collapsed)}>
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
                </div>
                {collapsed === false && (
                    <>
                        <Joints />
                        <CompoundMovements />
                    </>
                )}
            </Flex>
        </View>
    );
};
