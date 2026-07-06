import { useRef, type FormEvent } from 'react';

import { Button, Divider, Flex, Form, Heading, Icon, Item, Picker, TextField } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { SchemaRobotType } from '../robot-types';
import { SO101FormFields } from './catalog/so101';
import { WidowxAIFormFields } from './catalog/widowxai';
import { BiManualWidowxAIFormFields } from './catalog/widowxai-bimanual';
import { useRobotForm, useRobotFormFields, useSetRobotForm } from './provider';
import { SubmitNewRobotButton } from './submit-new-robot-button';

const RobotType = () => {
    const { activeType } = useRobotForm();
    const { setActiveType } = useSetRobotForm();

    return (
        <Picker
            isRequired
            label='Robot type'
            width='100%'
            selectedKey={activeType}
            onSelectionChange={(selected) => {
                if (selected !== null) {
                    setActiveType(selected as SchemaRobotType);
                }
            }}
        >
            <Item key={'SO101_Follower'}>SO101 Follower</Item>
            <Item key={'SO101_Leader'}>SO101 Leader</Item>
            <Item key={'Trossen_WidowXAI_Follower'}>Trossen WidowX AI Follower</Item>
            <Item key={'Trossen_WidowXAI_Leader'}>Trossen WidowX AI Leader</Item>
            <Item key={'Trossen_Bimanual_WidowXAI_Follower'}>Trossen Bimanual WidowX AI Follower</Item>
            <Item key={'Trossen_Bimanual_WidowXAI_Leader'}>Trossen Bimanual WidowX AI Leader</Item>
        </Picker>
    );
};

const FormFields = ({ robotType }: { robotType: SchemaRobotType }) => {
    const { formData, updateField } = useRobotFormFields();

    let formFields = null;
    switch (robotType) {
        case 'SO101_Follower':
        case 'SO101_Leader':
            formFields = <SO101FormFields />;
            break;
        case 'Trossen_WidowXAI_Follower':
        case 'Trossen_WidowXAI_Leader':
            formFields = <WidowxAIFormFields />;
            break;
        case 'Trossen_Bimanual_WidowXAI_Leader':
        case 'Trossen_Bimanual_WidowXAI_Follower':
            formFields = <BiManualWidowxAIFormFields />;
            break;
    }

    return (
        <>
            <TextField
                isRequired
                label='Robot name'
                width='100%'
                onChange={(name) => {
                    updateField('name', name);
                }}
                value={formData.name}
            />
            {formFields}
        </>
    );
};

export const RobotForm = ({ heading = 'Add new robot', submitButton = <SubmitNewRobotButton /> }) => {
    const { project_id } = useProjectId();

    const { activeType } = useRobotForm();

    const submitContainerRef = useRef<HTMLDivElement>(null);

    // Make Enter behave like clicking the submit button. A disabled button ignores
    // the click, so validation still applies.
    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        submitContainerRef.current?.querySelector('button')?.click();
    };

    return (
        <Flex direction='column' gap='size-200'>
            <Flex alignItems={'center'} gap='size-200'>
                <Button
                    href={paths.project.robots.index({ project_id })}
                    variant='secondary'
                    UNSAFE_style={{ border: 'none' }}
                >
                    <Icon>
                        <ChevronLeft color='white' fill='white' />
                    </Icon>
                </Button>

                <Heading>{heading}</Heading>
            </Flex>
            <Divider orientation='horizontal' size='S' />
            {/* Prevent native form submission, which reloads the page and discards form state.
                Advancing to the next step is handled by the submit button's onPress. */}
            <Form onSubmit={handleSubmit}>
                <Flex direction='column' gap='size-200'>
                    <Flex direction='column' gap='size-200' width='100%'>
                        <RobotType />

                        <FormFields robotType={activeType} />
                    </Flex>
                    <Divider orientation='horizontal' size='S' />
                    <div ref={submitContainerRef}>{submitButton}</div>
                </Flex>
            </Form>
        </Flex>
    );
};
