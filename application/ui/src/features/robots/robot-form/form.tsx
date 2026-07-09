import { Button, Flex, Heading, Icon, Item, Picker, TextField } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { SchemaRobotType } from '../robot-types';
import { SO101FormFields } from './catalog/so101';
import { WidowxAIFormFields } from './catalog/widowxai';
import { BiManualWidowxAIFormFields } from './catalog/widowxai-bimanual';
import { useRobotForm, useRobotFormFields, useSetRobotForm } from './provider';

export const RobotType = () => {
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

export const FormFields = () => {
    const { formData, updateField, activeType } = useRobotFormFields();

    let formFields = null;
    switch (activeType) {
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

export const RobotFormHeading = ({ heading }: { heading: string }) => {
    const { project_id } = useProjectId();

    return (
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
    );
};
