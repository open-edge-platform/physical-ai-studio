import { Dispatch, SetStateAction, useState } from 'react';

import { ActionButton, Flex, Grid, Heading, minmax, repeat, Slider, Switch, View } from '@geti-ui/ui';
import { ChevronDownSmallLight } from '@geti-ui/ui/icons';
import { radToDeg } from 'three/src/math/MathUtils.js';

import { getRobotConnectionErrorTitle } from '../../../api/errors';
import { useRobotCatalogDefinitionQuery } from '../robot-catalog.hooks';
import { featureRange, useLoadModelQuery } from '../robot-models-context';
import { InlineAlert } from '../setup-wizard/shared/inline-alert';
import { useRobot, useRobotId } from '../use-robot';
import { useRobotObservations } from '../use-robot-observations';
import { useSynchronizeModelJoints } from '../use-synchronize-model-joints';

type JointState = {
    name: string;
    value: number;
    rangeMin: number;
    rangeMax: number;
};
type JointsState = Array<JointState>;

// Without `onChange` the slider only shows the joint.
const Joint = ({ joint, onChange }: { joint: JointState; onChange?: (value: number) => void }) => {
    // While dragging, show the target, not the arm still easing toward it.
    const [target, setTarget] = useState<number | null>(null);
    const value = target ?? joint.value;

    return (
        <li>
            <View backgroundColor={'gray-50'} padding='size-115' UNSAFE_style={{ borderRadius: '4px' }}>
                <Grid areas={['name value', 'slider slider']} gap='size-100'>
                    <div style={{ gridArea: 'name' }}>
                        <span>{joint.name}</span>
                    </div>
                    <div style={{ gridArea: 'value', display: 'flex', justifyContent: 'end' }}>
                        <span style={{ color: 'var(--energy-blue-light)' }}>{value.toFixed(2)}&deg;</span>
                    </div>
                    <Flex gridArea='slider' gap='size-200'>
                        <Slider
                            aria-label={joint.name}
                            value={value}
                            minValue={joint.rangeMin}
                            maxValue={joint.rangeMax}
                            step={(joint.rangeMax - joint.rangeMin) / 1000}
                            onChange={(next) => {
                                setTarget(next);
                                onChange?.(next);
                            }}
                            onChangeEnd={() => setTarget(null)}
                            flexGrow={1}
                            isDisabled={onChange === undefined}
                        />
                    </Flex>
                </Grid>
            </View>
        </li>
    );
};

const Joints = ({
    joints,
    onChange,
}: {
    joints: Array<JointState & { isControllable?: boolean }>;
    onChange?: (name: string, value: number) => void;
}) => {
    return (
        <ul>
            <Grid gap='size-50' columns={repeat('auto-fit', minmax('size-4600', '1fr'))}>
                {joints.map((joint) => {
                    return (
                        <Joint
                            key={joint.name}
                            joint={joint}
                            onChange={
                                onChange && joint.isControllable ? (value) => onChange(joint.name, value) : undefined
                            }
                        />
                    );
                })}
            </Grid>
        </ul>
    );
};

// Get the default stationary joint setting with min and max range based on the urdf model
const useModelJoints = (): JointsState => {
    const robot = useRobot();
    const { data: model } = useLoadModelQuery(robot.type);

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
const useRobotJointsState = () => {
    const robot = useRobot();
    const { data: model } = useLoadModelQuery(robot.type);
    const { data: definition } = useRobotCatalogDefinitionQuery(robot.type);

    const { project_id, robot_id } = useRobotId();
    const { joints, error, errorCode, sendAction } = useRobotObservations(project_id, robot_id, { control: true });
    useSynchronizeModelJoints(joints, robot);

    return {
        joints: joints.map((joint) => {
            const range = model && featureRange(joint.name, model, definition.joint_map, robot);
            // A joint without a known range only shows: sending it a target could exceed its limits.
            return { ...joint, rangeMin: range?.min ?? -180, rangeMax: range?.max ?? 180, isControllable: !!range };
        }),
        error,
        errorCode,
        sendAction,
    };
};

const EnabledJointControls = ({ isExpanded }: { isExpanded: boolean }) => {
    const { joints, error, errorCode, sendAction } = useRobotJointsState();

    if (error) {
        return (
            <InlineAlert variant='error'>
                <strong>{getRobotConnectionErrorTitle(errorCode)}</strong>
                <br />
                {error}
            </InlineAlert>
        );
    }

    if (isExpanded) {
        return <Joints joints={joints} onChange={sendAction} />;
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
            zIndex={1}
            backgroundColor={'gray-100'}
            padding='size-100'
            margin='size-400'
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

                    <Switch isEmphasized isSelected={isConnected} onChange={setIsConnected}>
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
