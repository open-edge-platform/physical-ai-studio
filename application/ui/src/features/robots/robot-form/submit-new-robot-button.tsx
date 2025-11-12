import { Button } from '@geti/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useRobotFormBody } from './provider';

export const SubmitNewRobotButton = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const addRobotMutation = $api.useMutation('post', '/api/projects/{project_id}/robots');

    const body = useRobotFormBody(uuidv4());

    return (
        <Button
            variant='accent'
            isPending={addRobotMutation.isPending}
            isDisabled={body === null}
            onPress={async () => {
                if (body === null) {
                    return;
                }

                await addRobotMutation.mutateAsync(
                    {
                        params: { path: { project_id } },
                        body,
                    },
                    {
                        onSuccess: ({}, { body: { id: robot_id } }) => {
                            navigate(paths.project.robots.show({ project_id, robot_id }));
                        },
                    }
                );
            }}
        >
            Add robot
        </Button>
    );
};
