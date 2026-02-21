import { Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti/ui';

import { RobotForm } from '../../features/robots/robot-form/form';
import { Preview } from '../../features/robots/robot-form/preview';
import { RobotFormProvider } from '../../features/robots/robot-form/provider';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const New = () => {
    return (
        <RobotModelsProvider>
            <RobotFormProvider>
                <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
                    <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                        <Suspense fallback={<CenteredLoading />}>
                            <RobotForm />
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
