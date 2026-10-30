import { useRef, useState } from 'react';

import { Button, Flex, Keyboard, ProgressCircle, Slider, Switch, Text, toast, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getRobotConnectionErrorTitle } from '../../../api/errors';
import { SchemaTeleopControlConfig } from '../../../api/openapi-spec';
import { formatKeyCombo, getEffectiveBindings } from '../../hotkeys/key-combo';
import { useHeldHotkey } from '../../hotkeys/use-hotkey';
import { useProjectId } from '../../projects/use-project';
import { RobotViewer, UnavailableRobotViewer } from '../../robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../robots/robot-types';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { useSynchronizeModelJoints } from '../../robots/use-synchronize-model-joints';
import { ERROR_TOAST, failureOf, useRuntimeSession, useSessionJoints } from '../runtime-session-provider';

/**
 * Teleoperation as the UI starts it. The leader is read at 100 Hz, each reading one command, and on every
 * start the follower takes 3 s to reach the leader instead of jumping to it.
 */
export const TELEOP_CONTROL = { type: 'teleop', hz: 100, grace_s: 2 } as const satisfies SchemaTeleopControlConfig;

/** Drive the follower from the leader, or hold it where it is. */
export const TeleopSwitch = () => {
    const { state, statuses, send } = useRuntimeSession();
    const [isPending, setIsPending] = useState(false);

    const setTeleop = async (teleoperate: boolean) => {
        setIsPending(true);
        const ack = await send({
            command: 'set_control',
            control: teleoperate ? TELEOP_CONTROL : null,
        });
        setIsPending(false);
        if (!ack.ok) {
            toast.negative(ack.error?.message ?? 'Could not switch teleoperation.', ERROR_TOAST);
        }
    };

    return (
        <Switch
            isEmphasized
            isSelected={state?.control?.type === 'teleop'}
            // A failed environment stays loaded until it is loaded again, with nothing to drive.
            isDisabled={isPending || failureOf(statuses, 'environment') !== null}
            onChange={(teleoperate) => void setTeleop(teleoperate)}
        >
            Teleoperate
        </Switch>
    );
};

/**
 * While held, the follower moves by the leader's movement divided by the slowdown, from where it is.
 * On letting go it is back on absolute teleop, whose grace period closes the gap to the leader. The
 * teleop control keeps running throughout, so the follower is never left undriven by the switch.
 */
export const RelativeTeleop = () => {
    const { state, statuses, send } = useRuntimeSession();
    const [slowdown, setSlowdown] = useState(2);
    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const combo = getEffectiveBindings(settings.hotkeys.bindings)['relative_teleop'];
    // Whether this hold switched to relative, so letting go only switches back what it switched.
    const active = useRef(false);

    const isTeleop = state?.control?.type === 'teleop';
    const isRelative = state?.control?.type === 'teleop' && state.control.slowdown != null;

    const switchTo = async (to: number | null) => {
        const ack = await send({ command: 'set_teleop_slowdown', slowdown: to });
        if (!ack.ok) {
            toast.negative(ack.error?.message ?? 'Could not switch teleoperation.', ERROR_TOAST);
        }
    };
    const press = () => {
        if (isTeleop && !active.current) {
            active.current = true;
            void switchTo(slowdown);
        }
    };
    const release = () => {
        if (active.current) {
            active.current = false;
            void switchTo(null);
        }
    };

    useHeldHotkey(combo, press, release, failureOf(statuses, 'environment') === null);

    return (
        <Flex direction='column' gap='size-100'>
            <Slider
                label='Slowdown'
                value={slowdown}
                onChange={setSlowdown}
                minValue={1}
                maxValue={4}
                step={0.25}
                getValueLabel={(value) => `1/${value}`}
                isDisabled={isRelative}
                width='100%'
            />
            <Button
                variant={isRelative ? 'accent' : 'secondary'}
                isDisabled={!isTeleop}
                onPressStart={press}
                onPressEnd={release}
            >
                <Text>Hold for relative</Text>
                <Keyboard>{formatKeyCombo(combo)}</Keyboard>
            </Button>
        </Flex>
    );
};

const LiveRobot = ({ robot }: { robot: AvailableSchemaRobot }) => {
    const joints = useSessionJoints();
    useSynchronizeModelJoints(joints, robot.type);

    return <RobotViewer robot={robot} />;
};

/** The session's follower. ``showTeleop`` puts the Teleoperate switch on it, for pages without a side bar. */
export const RobotCell = ({ follower_id, showTeleop = true }: { follower_id: string; showTeleop?: boolean }) => {
    const { project_id } = useProjectId();
    const { state, error, isLoading } = useRuntimeSession();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id: follower_id } },
    });
    if (isUnavailableRobot(robot)) {
        return <UnavailableRobotViewer robotType={robot.type} />;
    }

    if (error !== null && !isLoading) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <InlineAlert variant='error'>
                    <strong>{getRobotConnectionErrorTitle(error.error_code)}</strong>
                    <br />
                    {error.message}
                </InlineAlert>
            </Flex>
        );
    }

    if (state?.environment?.robot_id !== follower_id) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <ProgressCircle isIndeterminate aria-label='Connecting robot' />
            </Flex>
        );
    }

    return (
        <RobotModelsProvider>
            <View
                minWidth='size-4000'
                minHeight='size-4000'
                width='100%'
                height='100%'
                backgroundColor={'gray-600'}
                position={'relative'}
            >
                <LiveRobot robot={robot} />
                {showTeleop && state.environment.leader_id !== null && (
                    <View position={'absolute'} right={0} top={0} padding='size-100'>
                        <TeleopSwitch />
                    </View>
                )}
            </View>
        </RobotModelsProvider>
    );
};
