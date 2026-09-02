import type { FormEvent } from 'react';

import { Button, Divider, Flex, Form, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { EnvironmentFormFields, EnvironmentFormHeading } from './form';
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

export const CreateEnvironmentForm = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const addEnvironmentMutation = $api.useMutation('post', '/api/projects/{project_id}/environments', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/environments', { params: { path: { project_id } } }]],
        },
    });

    const body = useEnvironmentFormBody(uuidv4());
    const isDisabled = useIsDisabled(body);

    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();

        if (isDisabled) {
            return;
        }

        addEnvironmentMutation.mutate(
            {
                params: { path: { project_id } },
                body,
            },
            {
                onSuccess: (createdEnvironment) => {
                    navigate(paths.project.environments.show({ project_id, environment_id: createdEnvironment.id }));
                },
            }
        );
    };

    return (
        <Flex direction='column' gap='size-200'>
            <EnvironmentFormHeading heading='Add new environment' />
            <Divider size='S' />
            <Form onSubmit={handleSubmit}>
                <Flex gap='size-200' alignItems='end' direction='column'>
                    <EnvironmentFormFields />
                    <Divider orientation='horizontal' size='S' />
                    <View>
                        <Button
                            variant='accent'
                            type='submit'
                            isDisabled={isDisabled}
                            isPending={addEnvironmentMutation.isPending}
                        >
                            Add environment
                        </Button>
                    </View>
                </Flex>
            </Form>
        </Flex>
    );
};
