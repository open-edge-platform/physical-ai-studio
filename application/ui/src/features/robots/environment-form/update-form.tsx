import type { FormEvent } from 'react';

import { Button, Divider, Flex, Form, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { useEnvironmentId } from '../use-environment';
import { EnvironmentFormFields, EnvironmentFormHeading } from './form';
import { useEnvironmentFormBody } from './provider';

export const UpdateEnvironmentForm = () => {
    const navigate = useNavigate();
    const { project_id, environment_id } = useEnvironmentId();

    const updateEnvironmentMutation = $api.useMutation(
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
    const isDisabled = body.name.length === 0 || body.robots.length === 0 || body.cameras.length === 0;

    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();

        if (isDisabled) {
            return;
        }

        updateEnvironmentMutation.mutate(
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
    };

    return (
        <Flex direction='column' gap='size-200'>
            <EnvironmentFormHeading heading='Update environment' />
            <Divider orientation='horizontal' size='S' />
            <Form onSubmit={handleSubmit}>
                <Flex gap='size-200' alignItems='end' direction='column'>
                    <EnvironmentFormFields />
                    <Divider orientation='horizontal' size='S' />
                    <View>
                        <Button
                            variant='accent'
                            type='submit'
                            isDisabled={isDisabled}
                            isPending={updateEnvironmentMutation.isPending}
                        >
                            Update environment
                        </Button>
                    </View>
                </Flex>
            </Form>
        </Flex>
    );
};
