import { Button } from '@geti-ui/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useEnvironmentFormBody } from './provider';

const useIsDisabled = (body: ReturnType<typeof useEnvironmentFormBody>) => {
    const { project_id } = useProjectId();
    const robotsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
        params: { path: { project_id } },
    });

    if (body.name.trim().length === 0) {
        return true;
    }

    if (robotsQuery.data.length > 0 && body.robots.length === 0) {
        return true;
    }

    return false;
};

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
    const isDisabled = useIsDisabled(body);

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
