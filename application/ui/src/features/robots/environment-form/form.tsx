import { Button, Divider, Flex, Form, Heading, Icon, Text, TextField, View } from '@geti/ui';
import { ChevronLeft } from '@geti/ui/icons';

import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { CameraForm } from './camera-form';
import { useEnvironmentForm, useSetEnvironmentForm } from './provider';
import { RobotForm } from './robot-form';
import { SubmitNewEnvironmentButton } from './submit-new-environment-button';

export const EnvironmentForm = ({ heading = 'Add new environment', submitButton = <SubmitNewEnvironmentButton /> }) => {
    const { project_id } = useProjectId();
    const environmentForm = useEnvironmentForm();
    const setEnvironmentForm = useSetEnvironmentForm();

    return (
        <Flex direction='column' gap='size-200'>
            <Flex alignItems={'center'} gap='size-200'>
                <Button
                    href={paths.project.environments.index({ project_id })}
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
            <Form>
                <Flex gap='size-200' alignItems='end' direction={'column'}>
                    <View maxWidth='size-5000' alignSelf={'start'}>
                        <Text
                            UNSAFE_style={{
                                color: 'var(--spectrum-global-color-gray-700)',
                            }}
                        >
                            Recording datasets is based on an environment setup that includes robots and cameras. A
                            single environment setup represents your physical setup that you use for tele operating the
                            robot.
                        </Text>
                    </View>
                    <TextField
                        isRequired
                        necessityIndicator='label'
                        label='name'
                        width='100%'
                        onChange={(name) => {
                            setEnvironmentForm((oldForm) => {
                                return { ...oldForm, name };
                            });
                        }}
                        value={environmentForm.name}
                    />

                    <Divider size='S' />

                    <RobotForm />

                    <Divider size='S' />

                    <CameraForm />
                    <Divider orientation='horizontal' size='S' />
                    <View>{submitButton}</View>
                </Flex>
            </Form>
        </Flex>
    );
};
