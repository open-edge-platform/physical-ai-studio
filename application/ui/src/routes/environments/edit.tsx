import { Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti/ui';

import { EnvironmentForm } from '../../features/robots/environment-form/form';
import { Preview } from '../../features/robots/environment-form/preview';
import {
    EnvironmentFormProvider,
    EnvironmentFormState,
    RobotConfiguration,
} from '../../features/robots/environment-form/provider';
import { UpdateEnvironmentButton } from '../../features/robots/environment-form/update-environment-button';
import { useEnvironment } from '../../features/robots/use-environment';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const Edit = () => {
    const environment = useEnvironment();

    const environmentForm: EnvironmentFormState = {
        name: environment.name,
        camera_ids: environment.cameras?.map(({ id }) => id!) ?? [],
        robots:
            environment.robots?.map((robot): RobotConfiguration => {
                return {
                    robot_id: robot.robot.id,
                    teleoperator:
                        robot.tele_operator.type === 'robot'
                            ? {
                                  type: 'robot',
                                  robot_id: robot.tele_operator.robot_id,
                              }
                            : { type: 'none' },
                };
            }) ?? [],
    };

    return (
        <EnvironmentFormProvider environment={environmentForm}>
            <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
                <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                    <Suspense fallback={<CenteredLoading />}>
                        <EnvironmentForm heading='Update environment' submitButton={<UpdateEnvironmentButton />} />
                    </Suspense>
                </View>
                <View gridArea='controls' backgroundColor={'gray-50'}>
                    <Preview />
                </View>
            </Grid>
        </EnvironmentFormProvider>
    );
};
