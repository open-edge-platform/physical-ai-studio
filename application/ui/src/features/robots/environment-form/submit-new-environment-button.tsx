import { Button } from '@geti-ui/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useEnvironmentFormBody } from './provider';

export const SubmitNewEnvironmentButton = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const addEnvironmentMutation = $api.useMutation('post', '/api/projects/{project_id}/environments', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/environments', { params: { path: { project_id } } }]],
        },
    });

    const environment_id = uuidv4();
    const body = useEnvironmentFormBody(environment_id);
    const isDisabled = false; // body.name.length === 0 || body.robots.length === 0 || body.camera_ids.length === 0;

    return (
        <Button
            variant='accent'
            isPending={addEnvironmentMutation.isPending}
            isDisabled={isDisabled}
            onPress={async () => {
                if (isDisabled) {
                    return;
                }

                await addEnvironmentMutation.mutateAsync(
                    {
                        params: { path: { project_id } },
                        body,
                    },
                    {
                        onSuccess: ({}, { body: {} }) => {
                            navigate(paths.project.environments.show({ project_id, environment_id }));
                        },
                    }
                );
            }}
        >
            Add environment
        </Button>
    );
};
