import { Button } from '@geti-ui/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useRobotForm, useRobotFormBody } from './provider';

export const SubmitNewRobotButton = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const addRobotMutation = $api.useMutation('post', '/api/projects/{project_id}/robots', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/robots', { params: { path: { project_id } } }],
                ['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }],
            ],
        },
    });

    const body = useRobotFormBody(uuidv4());

    const { activeType } = useRobotForm();
    const isSO101 = activeType === 'SO101_Follower' || activeType === 'SO101_Leader';

    return (
        <Button
            variant='accent'
            isPending={addRobotMutation.isPending}
            isDisabled={body === null}
            onPress={async () => {
                if (body === null) {
                    return;
                }

                if (isSO101) {
                    return navigate(paths.project.robots.so101Setup({ project_id }));
                }

                await addRobotMutation.mutateAsync(
                    {
                        params: { path: { project_id } },
                        body,
                    },
                    {
                        onSuccess: (createdRobot) => {
                            navigate(paths.project.robots.show({ project_id, robot_id: createdRobot.id }));
                        },
                    }
                );
            }}
        >
            {isSO101 ? 'Begin setup' : 'Add robot'}
        </Button>
    );
};
