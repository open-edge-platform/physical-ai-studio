import { Dispatch, SetStateAction, useState } from 'react';

import { ActionButton, Flex, Grid, Heading, minmax, repeat, Slider, Switch, View } from '@geti-ui/ui';
import { ChevronDownSmallLight } from '@geti-ui/ui/icons';
import { radToDeg } from 'three/src/math/MathUtils.js';

import { urdfPathForType, useRobotModels } from '../robot-models-context';
import { useJointState, useSynchronizeModelJoints } from '../use-joint-state';
import { useRobot, useRobotId } from '../use-robot';

type JointState = {
    name: string;
    value: number;
    rangeMin: number;
    rangeMax: number;
};
type JointsState = Array<JointState>;

const Joint = ({ joint }: { joint: JointState }) => {
    return (
        <li>
            <View backgroundColor={'gray-50'} padding='size-115' UNSAFE_style={{ borderRadius: '4px' }}>
                <Grid areas={['name value', 'slider slider']} gap='size-100'>
                    <div style={{ gridArea: 'name' }}>
                        <span>{joint.name}</span>
                    </div>
                    <div style={{ gridArea: 'value', display: 'flex', justifyContent: 'end' }}>
                        <span style={{ color: 'var(--energy-blue-light)' }}>{joint.value.toFixed(2)}&deg;</span>
                    </div>
                    <Flex gridArea='slider' gap='size-200'>
                        <Slider
                            aria-label={joint.name}
                            value={joint.value}
                            minValue={joint.rangeMin}
                            maxValue={joint.rangeMax}
                            flexGrow={1}
                            isDisabled={true}
                        />
                    </Flex>
                </Grid>
            </View>
        </li>
    );
};

const Joints = ({ joints }: { joints: JointsState }) => {
    return (
        <ul>
            <Grid gap='size-50' columns={repeat('auto-fit', minmax('size-4600', '1fr'))}>
                {joints.map((joint) => {
                    return <Joint key={joint.name} joint={joint} />;
                })}
            </Grid>
        </ul>
    );
};

// Get the default stationary joint setting with min and max range based on the urdf model
const useModelJoints = (): JointsState => {
    const { getModel } = useRobotModels();

    const robot = useRobot();
    const model = getModel(urdfPathForType(robot.type));

    const modelJoints = Object.values(model?.joints ?? {});
    const joints: JointsState = modelJoints
        .filter((joint) => joint.jointType !== 'fixed')
        .map((joint) => {
            const rangeMax = radToDeg(joint.limit.upper);
            const rangeMin = radToDeg(joint.limit.lower);

            return { name: joint.name, value: 0, rangeMin, rangeMax };
        })
        .toReversed();

    return joints;
};

// Combine the joint range of the urdf model with actual joint state from robot
const useRobotJointsState = (): JointsState => {
    const robot = useRobot();
    const modelJoints = useModelJoints();

    const { project_id, robot_id } = useRobotId();
    const { joints } = useJointState(project_id, robot_id);
    useSynchronizeModelJoints(joints, robot.type);

    return joints.map((joint) => {
        const modelJoint = modelJoints.find(({ name }) => name === joint.name);
        const rangeMax = modelJoint === undefined ? 180 : radToDeg(modelJoint.rangeMax);
        const rangeMin = modelJoint === undefined ? -180 : radToDeg(modelJoint.rangeMin);

        return { ...joint, rangeMin, rangeMax };
    });
};

const EnabledJointControls = ({ isExpanded }: { isExpanded: boolean }) => {
    const joints = useRobotJointsState();

    if (isExpanded) {
        return <Joints joints={joints} />;
    }

    return null;
};

const DisabledJointsControls = ({ isExpanded }: { isExpanded: boolean }) => {
    const joints: JointsState = useModelJoints();

    if (isExpanded) {
        return <Joints joints={joints} />;
    }

    return null;
};

export const JointControls = ({
    isConnected,
    setIsConnected,
}: {
    isConnected: boolean;
    setIsConnected: Dispatch<SetStateAction<boolean>>;
}) => {
    const [isExpanded, setIsExpanded] = useState(true);

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
                <Flex justifyContent={'space-between'}>
                    <ActionButton onPress={() => setIsExpanded((c) => !c)}>
                        <Heading level={4} marginX='size-100'>
                            <Flex alignItems='center' gap='size-100'>
                                <ChevronDownSmallLight
                                    fill='white'
                                    style={{
                                        transform: isExpanded ? 'rotate(180deg)' : '',
                                        animation: 'transform ease-in-out 0.1s',
                                    }}
                                />
                                Joint state
                            </Flex>
                        </Heading>
                    </ActionButton>

                    <Switch isSelected={isConnected} onChange={setIsConnected}>
                        Connect
                    </Switch>
                </Flex>
                {isConnected ? (
                    <EnabledJointControls isExpanded={isExpanded} />
                ) : (
                    <DisabledJointsControls isExpanded={isExpanded} />
                )}
            </Flex>
        </View>
    );
};
