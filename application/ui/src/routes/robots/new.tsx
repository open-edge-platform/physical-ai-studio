import { Suspense } from 'react';

import { Button, Flex, Grid, Loading, minmax, View } from '@geti/ui';
import { useNavigate } from 'react-router';

import { RobotForm } from '../../features/robots/robot-form/form';
import { Preview } from '../../features/robots/robot-form/preview';
import { RobotFormProvider, useRobotForm } from '../../features/robots/robot-form/provider';
import { SubmitNewRobotButton } from '../../features/robots/robot-form/submit-new-robot-button';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { paths } from '../../router';
import { useProjectId } from '../../features/projects/use-project';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

/**
 * Submit button that adapts to the selected robot type:
 * - SO101 types: navigates to the setup wizard route with form values as search params
 * - Other types (Trossen): directly creates the robot via POST (default behavior)
 */
const NewRobotSubmitButton = () => {
    const robotForm = useRobotForm();
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const isSO101 = robotForm.type?.toLowerCase().startsWith('so101') ?? false;

    if (!isSO101) {
        return <SubmitNewRobotButton />;
    }

    const isDisabled = !robotForm.name || !robotForm.type || !robotForm.serial_number;

    return (
        <Button
            variant='accent'
            isDisabled={isDisabled}
            onPress={() => {
                const params = new URLSearchParams({
                    name: robotForm.name,
                    type: robotForm.type,
                    serial_number: robotForm.serial_number ?? '',
                    connection_string: robotForm.connection_string ?? '',
                });

                navigate(
                    `${paths.project.robots.so101Setup({ project_id })}?${params.toString()}`
                );
            }}
        >
            Begin Setup
        </Button>
    );
};

export const New = () => {
    return (
        <RobotModelsProvider>
            <RobotFormProvider>
                <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
                    <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                        <Suspense fallback={<CenteredLoading />}>
                            <RobotForm submitButton={<NewRobotSubmitButton />} />
                        </Suspense>
                    </View>
                    <View gridArea='controls' backgroundColor={'gray-50'} padding='size-400'>
                        <Preview />
                    </View>
                </Grid>
            </RobotFormProvider>
        </RobotModelsProvider>
    );
};
