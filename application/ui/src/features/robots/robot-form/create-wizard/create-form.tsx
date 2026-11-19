import { Button, Divider, Flex, Form, Heading } from '@geti-ui/ui';

import { useRobotCatalogQuery } from '../../robot-catalog.hooks';
import { FormFields, RobotFormHeading } from '../form';
import { useRobotForm } from '../provider';
import { RoleBadge } from './role-badge';
import { useSubmitRobot } from './use-submit-robot';

interface CreateRobotFormProps {
    onChangeRobot: () => void;
}

export const CreateRobotForm = ({ onChangeRobot }: CreateRobotFormProps) => {
    const { activeType } = useRobotForm();
    const { data: catalog } = useRobotCatalogQuery();
    const submit = useSubmitRobot();
    const entry = catalog.find(({ type }) => type === activeType);

    return (
        <Flex direction='column' gap='size-200'>
            <RobotFormHeading heading='Configure new robot' />
            <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                <Flex direction='column' alignItems='start' gap='size-50'>
                    <Heading level={3} margin={0}>
                        {entry?.display_name}
                    </Heading>
                    {entry && <RoleBadge role={entry.role} />}
                </Flex>
                <Button variant='secondary' onPress={onChangeRobot}>
                    Change
                </Button>
            </Flex>
            <Divider orientation='horizontal' size='S' />
            <Form onSubmit={submit.handleSubmit}>
                <Flex direction='column' gap='size-200'>
                    <FormFields />
                    <Divider orientation='horizontal' size='S' />
                    <Flex>
                        <Button
                            variant='accent'
                            type='submit'
                            isDisabled={submit.isDisabled}
                            isPending={submit.isPending}
                        >
                            {submit.label}
                        </Button>
                    </Flex>
                </Flex>
            </Form>
        </Flex>
    );
};
