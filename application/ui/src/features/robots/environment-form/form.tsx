import { Divider, Text, TextField, View } from '@geti-ui/ui';

import { FormHeading } from '../../../components/form-heading/form-heading';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { CameraForm } from './camera-form';
import { useEnvironmentForm, useSetEnvironmentForm } from './provider';
import { RobotForm } from './robot-form';

interface EnvironmentFormHeadingProps {
    heading: string;
}

export const EnvironmentFormHeading = ({ heading }: EnvironmentFormHeadingProps) => {
    const { project_id } = useProjectId();

    return (
        <FormHeading
            heading={heading}
            backTo={paths.project.environments.index({ project_id })}
            backLabel='Back to environments'
        />
    );
};

export const EnvironmentFormFields = () => {
    const environmentForm = useEnvironmentForm();
    const setEnvironmentForm = useSetEnvironmentForm();

    return (
        <>
            <View maxWidth='size-5000' alignSelf='start'>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>
                    Recording datasets is based on an environment setup that includes robots and cameras. A single
                    environment setup represents your physical setup that you use for tele operating the robot.
                </Text>
            </View>

            <TextField
                // eslint-disable-next-line jsx-a11y/no-autofocus
                autoFocus
                isRequired
                label='Name'
                width='100%'
                value={environmentForm.name}
                onChange={(name) => {
                    setEnvironmentForm((oldForm) => {
                        return { ...oldForm, name };
                    });
                }}
            />

            <Divider size='S' />

            <RobotForm />

            <Divider size='S' />

            <CameraForm />
        </>
    );
};
