import { View } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { useProjectId } from '../../../projects/use-project';
import { RobotViewer } from '../../controller/robot-viewer';
import { RobotModelsProvider } from '../../robot-models-context';
import { useJointState, useSynchronizeModelJoints } from '../../use-joint-state';

const InnerCell = ({ follower_id, leader_id }: { follower_id: string; leader_id?: string }) => {
    const { project_id } = useProjectId();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id: follower_id } },
    });

    const { joints } = useJointState(project_id, follower_id, leader_id);
    useSynchronizeModelJoints(joints, robot.type);

    return (
        <View minWidth='size-4000' minHeight='size-4000' width='100%' height='100%' backgroundColor={'gray-600'}>
            <RobotViewer robot={robot} />
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
