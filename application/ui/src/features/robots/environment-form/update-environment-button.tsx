import { Button } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { useEnvironmentId } from '../use-environment';
import { useEnvironmentFormBody } from './provider';

export const UpdateEnvironmentButton = () => {
    const navigate = useNavigate();
    const { project_id, environment_id } = useEnvironmentId();

    const updateEnvironmentsMutation = $api.useMutation(
        'put',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            meta: {
                invalidates: [
                    ['get', '/api/projects/{project_id}/environments', { params: { path: { project_id } } }],
                    [
                        'get',
                        '/api/projects/{project_id}/environments/{environment_id}',
                        { params: { path: { project_id, environment_id } } },
                    ],
                ],
            },
        }
    );
    const body = useEnvironmentFormBody(environment_id);
    const isDisabled = body.name.length === 0 || body.robots.length === 0 || body.camera_ids.length === 0;

    return (
        <Button
            variant='accent'
            isPending={updateEnvironmentsMutation.isPending}
            isDisabled={isDisabled}
            onPress={async () => {
                if (isDisabled) {
                    return;
                }

                await updateEnvironmentsMutation.mutateAsync(
                    {
                        params: { path: { project_id, environment_id } },
                        body,
                    },
                    {
                        onSuccess: () => {
                            navigate(paths.project.environments.show({ project_id, environment_id }));
                        },
                    }
                );
            }}
        >
            Update environment
        </Button>
    );
};
