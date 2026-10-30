import { useEffect } from 'react';

import { Button, Flex, ProgressCircle, Switch, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';
import { Focusable } from 'react-aria';

import { $api } from '../../../../api/client';
import { getRobotConnectionErrorTitle } from '../../../../api/errors';
import { useProjectId } from '../../../projects/use-project';
import { useRobotSimulationLaunch } from '../../../simulation/robot-simulation';
import { useSimulationDialog } from '../../../simulation/simulation-dialog';
import { SimulationNotRunning } from '../../../simulation/simulation-not-running';
import { SimulationViewer } from '../../../simulation/simulation-viewer';
import { useSimulation } from '../../../simulation/use-simulation';
import { RobotViewer, UnavailableRobotViewer } from '../../controller/robot-viewer';
import { RobotModelsProvider } from '../../robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../robot-types';
import { InlineAlert } from '../../setup-wizard/shared/inline-alert';
import { useJointState, useSynchronizeModelJoints } from '../../use-joint-state';

/** Reconnect a session that failed for want of a simulation once Studio's simulation for the robot runs. */
const useRestartWhenSimulationRuns = (robotId: string, errorCode: string | null, restart: () => void) => {
    const { data: simulation } = useSimulation();
    const isRunning = simulation?.robot_id === robotId && simulation.status === 'running';

    useEffect(() => {
        if (isRunning && errorCode === 'simulation_not_running') {
            restart();
        }
    }, [isRunning, errorCode, restart]);
};

const AvailableRobotCell = ({
    robot,
    followerId,
    leaderId,
    cameraIds,
}: {
    robot: AvailableSchemaRobot;
    followerId: string;
    leaderId?: string;
    cameraIds: string[];
}) => {
    const { project_id } = useProjectId();
    const jointState = useJointState(project_id, followerId, leaderId, cameraIds);
    useRestartWhenSimulationRuns(robot.id, jointState.errorCode, jointState.restart);
    // Rendered beside the cell's content, not in it: the "not running" view goes away once the
    // simulation runs, and the dialog has to stay to show that it does.
    const simulationDialog = useSimulationDialog(robot);

    return (
        <>
            <RobotCellContent
                robot={robot}
                leaderId={leaderId}
                onStartSimulation={simulationDialog.open}
                {...jointState}
            />
            {simulationDialog.dialog}
        </>
    );
};

const RobotCellContent = ({
    robot,
    leaderId,
    joints,
    state,
    error,
    errorCode,
    warning,
    setFollowerSource,
    restart,
    onStartSimulation,
}: {
    robot: AvailableSchemaRobot;
    leaderId?: string;
    onStartSimulation?: () => void;
} & ReturnType<typeof useJointState>) => {
    useSynchronizeModelJoints(joints, robot.type);
    const isSimulated = useRobotSimulationLaunch(robot.type) !== null;

    const canTeleoperate = state.has_leader;

    const isTeleoperating = state.follower_source === 'teleop';

    if (error && errorCode === 'simulation_not_running') {
        return <SimulationNotRunning robot={robot} onStart={onStartSimulation} onReconnect={restart} />;
    }

    if (error) {
        return (
            <View width='100%' height='100%' padding='size-200'>
                <Flex
                    width='100%'
                    height='100%'
                    justifyContent='center'
                    alignItems='center'
                    direction='column'
                    gap='size-100'
                >
                    <InlineAlert variant='error'>
                        <strong>{getRobotConnectionErrorTitle(errorCode)}</strong>
                        <br />
                        {error}
                    </InlineAlert>
                    {errorCode === 'runtime_session_busy' && (
                        <Button variant='primary' onPress={restart}>
                            Restart session
                        </Button>
                    )}
                </Flex>
            </View>
        );
    }

    if (!state.connected) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <ProgressCircle isIndeterminate />
            </Flex>
        );
    }

    const controls = (
        <Flex gap='size-100' alignItems='center'>
            <Button variant='secondary' onPress={restart}>
                Restart session
            </Button>
            {leaderId !== undefined && (
                <TooltipTrigger delay={300}>
                    {/* Disabled elements don't fire hover events, so the tooltip trigger is
                        moved to a wrapping span (react-aria's documented workaround) rather
                        than the Switch itself. */}
                    <Focusable excludeFromTabOrder>
                        <span>
                            <Switch
                                isEmphasized
                                isSelected={isTeleoperating}
                                onChange={(b) => setFollowerSource(b ? 'teleop' : 'hold')}
                                isDisabled={!canTeleoperate}
                            >
                                Teleoperate
                            </Switch>
                        </span>
                    </Focusable>
                    <Tooltip>
                        {canTeleoperate
                            ? 'Control the follower using the leader robot'
                            : 'Connect a leader robot to enable teleoperation'}
                    </Tooltip>
                </TooltipTrigger>
            )}
        </Flex>
    );

    return (
        <View
            minWidth='size-4000'
            minHeight='size-4000'
            width='100%'
            height='100%'
            backgroundColor={'gray-600'}
            position={'relative'}
        >
            {isSimulated ? <SimulationViewer robot={robot} actions={controls} /> : <RobotViewer robot={robot} />}
            {warning && (
                <View
                    position={'absolute'}
                    left={0}
                    top={isSimulated ? 'size-600' : 0}
                    padding='size-100'
                    maxWidth='size-4600'
                >
                    <InlineAlert variant='warning'>{warning}</InlineAlert>
                </View>
            )}
            {isSimulated ? null : (
                <View position={'absolute'} right={0} top={0} padding='size-100'>
                    {controls}
                </View>
            )}
        </View>
    );
};

export const RobotCell = ({
    follower_id,
    leader_id,
    camera_ids,
}: {
    follower_id: string;
    leader_id?: string;
    camera_ids: string[];
}) => {
    const { project_id } = useProjectId();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id: follower_id } },
    });
    if (isUnavailableRobot(robot)) {
        return <UnavailableRobotViewer robotType={robot.type} />;
    }

    return (
        <RobotModelsProvider>
            <AvailableRobotCell robot={robot} followerId={follower_id} leaderId={leader_id} cameraIds={camera_ids} />
        </RobotModelsProvider>
    );
};
