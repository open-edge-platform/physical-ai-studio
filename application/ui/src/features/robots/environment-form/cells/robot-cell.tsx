import { Flex, ProgressCircle, Switch, View } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { useProjectId } from '../../../projects/use-project';
import { RobotViewer } from '../../controller/robot-viewer';
import { RobotModelsProvider } from '../../robot-models-context';
import { RobotActionReadState, useJointState, useSynchronizeModelJoints } from '../../use-joint-state';

const InnerCell = ({ follower_id, leader_id }: { follower_id: string; leader_id?: string }) => {
    const { project_id } = useProjectId();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id: follower_id } },
    });

    const { joints, state, setFollowerSource } = useJointState(project_id, follower_id, leader_id);
    useSynchronizeModelJoints(joints, robot.type);

    const isTeleoperating = state.follower_source === RobotActionReadState.Teleoperation;

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
            {leader_id !== undefined && (
                <View position={'absolute'} right={0} top={0}>
                    <Switch
                        isSelected={isTeleoperating}
                        onChange={(b) =>
                            setFollowerSource(b ? RobotActionReadState.Teleoperation : RobotActionReadState.None)
                        }
                    >
                        Teleoperate
                    </Switch>
                </View>
            )}
        </View>
    );
};

export const RobotCell = ({ follower_id, leader_id }: { follower_id: string; leader_id?: string }) => {
    return (
        <RobotModelsProvider>
            <InnerCell follower_id={follower_id} leader_id={leader_id} />
        </RobotModelsProvider>
    );
};
