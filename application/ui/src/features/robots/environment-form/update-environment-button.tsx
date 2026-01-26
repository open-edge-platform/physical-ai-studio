import { Button } from '@geti/ui';
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
        '/api/projects/{project_id}/environments/{environment_id}'
    );
    const body = useEnvironmentFormBody(environment_id);

    return (
        <Button
            variant='accent'
            isPending={updateEnvironmentsMutation.isPending}
            isDisabled={body === null}
            onPress={async () => {
                if (body === null) {
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
