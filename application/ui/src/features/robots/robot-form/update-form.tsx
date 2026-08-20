import type { FormEvent } from 'react';

import { Button, Divider, Flex, Form, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { useRobotId } from '../use-robot';
import { FormFields, RobotFormHeading } from './form';
import { useRobotFormBody } from './provider';

export const UpdateRobotForm = () => {
    const navigate = useNavigate();
    const { robot_id, project_id } = useRobotId();

    const updateRobotMutation = $api.useMutation('put', '/api/projects/{project_id}/robots/{robot_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/robots', { params: { path: { project_id } } }],
                ['get', '/api/projects/{project_id}/robots/{robot_id}', { params: { path: { project_id, robot_id } } }],
            ],
        },
    });

    const body = useRobotFormBody(robot_id);

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault();

        if (body === null) {
            return;
        }

        await updateRobotMutation.mutateAsync({
            params: { path: { project_id, robot_id } },
            body,
        });

        navigate(paths.project.robots.show({ project_id, robot_id }));
    };

    return (
        <Flex direction='column' gap='size-200'>
            <RobotFormHeading heading='Update robot' />
            <Divider orientation='horizontal' size='S' />
            <Form onSubmit={handleSubmit}>
                <Flex direction='column' gap='size-200'>
                    <Flex direction='column' gap='size-200' width='100%'>
                        <FormFields />
                    </Flex>
                    <Divider orientation='horizontal' size='S' />
                    <View>
                        <Button
                            variant='accent'
                            type='submit'
                            isDisabled={body === null}
                            isPending={updateRobotMutation.isPending}
                        >
                            Update robot
                        </Button>
                    </View>
                </Flex>
            </Form>
        </Flex>
    );
};
