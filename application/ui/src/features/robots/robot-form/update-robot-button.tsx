import { Button } from '@geti/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { useRobotId } from '../use-robot';
import { useRobotFormBody } from './provider';

export const UpdateRobotButton = () => {
    const navigate = useNavigate();
    const { project_id, robot_id } = useRobotId();

    const updateRobotMutation = $api.useMutation('put', '/api/projects/{project_id}/robots/{robot_id}');
    const body = useRobotFormBody(robot_id);

    return (
        <Button
            variant='accent'
            isPending={updateRobotMutation.isPending}
            isDisabled={body === null}
            onPress={async () => {
                if (body === null) {
                    return;
                }

                await updateRobotMutation.mutateAsync(
                    {
                        params: { path: { project_id, robot_id } },
                        body,
                    },
                    {
                        onSuccess: () => {
                            navigate(paths.project.robots.show({ project_id, robot_id }));
                        },
                    }
                );
            }}
        >
            Update robot
        </Button>
    );
};
