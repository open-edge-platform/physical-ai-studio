import { Divider, Flex, Heading, Text, TextField, View } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';
import { Link } from 'react-router';

import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { CameraForm } from './camera-form';
import { useEnvironmentForm, useSetEnvironmentForm } from './provider';
import { RobotForm } from './robot-form';

import classes from './form.module.css';

interface EnvironmentFormHeadingProps {
    heading: string;
}

export const EnvironmentFormHeading = ({ heading }: EnvironmentFormHeadingProps) => {
    const { project_id } = useProjectId();

    return (
        <Flex alignItems='center' gap='size-200'>
            <Link
                className={classes.link}
                aria-label='Back to environments'
                to={paths.project.environments.index({ project_id })}
            >
                <ChevronLeft color='white' fill='white' />
            </Link>
            <Heading>{heading}</Heading>
        </Flex>
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
