import type { FormEvent } from 'react';

import { Button, Divider, Flex, Form, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { FormFields, RobotFormHeading, RobotType } from './form';
import { useRobotForm, useRobotFormBody } from './provider';

export const CreateRobotForm = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();
    const { activeType } = useRobotForm();

    const addRobotMutation = $api.useMutation('post', '/api/projects/{project_id}/robots', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/robots', { params: { path: { project_id } } }],
                ['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }],
            ],
        },
    });

    const body = useRobotFormBody(uuidv4());
    const isSO101 = activeType === 'SO101_Follower' || activeType === 'SO101_Leader';

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault();

        if (body === null) {
            return;
        }

        if (isSO101) {
            navigate(paths.project.robots.so101Setup({ project_id }));
            return;
        }

        const createdRobot = await addRobotMutation.mutateAsync({
            params: { path: { project_id } },
            body,
        });

        navigate(paths.project.robots.show({ project_id, robot_id: createdRobot.id }));
    };

    const isCreateDisabled = body === null;

    return (
        <Flex direction='column' gap='size-200'>
            <RobotFormHeading heading='Add new robot' />
            <Divider orientation='horizontal' size='S' />
            <Form onSubmit={handleSubmit}>
                <Flex direction='column' gap='size-200'>
                    <Flex direction='column' gap='size-200' width='100%'>
                        <RobotType />
                        <FormFields />
                    </Flex>
                    <Divider orientation='horizontal' size='S' />
                    <View>
                        <Button
                            variant='accent'
                            type='submit'
                            isDisabled={isCreateDisabled}
                            isPending={addRobotMutation.isPending}
                        >
                            {isSO101 ? 'Begin Setup' : 'Add robot'}
                        </Button>
                    </View>
                </Flex>
            </Form>
        </Flex>
    );
};
