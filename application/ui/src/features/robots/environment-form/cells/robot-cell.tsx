import { Button, Flex, ProgressCircle, Switch, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';
import { Focusable } from 'react-aria';

import { $api } from '../../../../api/client';
import { getRobotConnectionErrorTitle } from '../../../../api/errors';
import { useProjectId } from '../../../projects/use-project';
import { RobotViewer, UnavailableRobotViewer } from '../../controller/robot-viewer';
import { RobotModelsProvider } from '../../robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../robot-types';
import { InlineAlert } from '../../setup-wizard/shared/inline-alert';
import { useJointState, useSynchronizeModelJoints } from '../../use-joint-state';

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
    const { joints, state, error, errorCode, warning, setFollowerSource, restart } = useJointState(
        project_id,
        followerId,
        leaderId,
        cameraIds
    );
    useSynchronizeModelJoints(joints, robot.type);

    const canTeleoperate = state.has_leader;

    const isTeleoperating = state.follower_source === 'teleop';

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

    return (
        <View
            minWidth='size-4000'
            minHeight='size-4000'
            width='100%'
            height='100%'
            backgroundColor={'gray-600'}
            position={'relative'}
        >
            <RobotViewer robot={robot} />
            {warning && (
                <View position={'absolute'} left={0} top={0} padding='size-100' maxWidth='size-4600'>
                    <InlineAlert variant='warning'>{warning}</InlineAlert>
                </View>
            )}
            <View position={'absolute'} right={0} top={0} padding='size-100'>
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
            </View>
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
