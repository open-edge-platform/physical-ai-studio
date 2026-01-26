import { Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti/ui';

import { EnvironmentForm } from '../../features/robots/environment-form/form';
import { Preview } from '../../features/robots/environment-form/preview';
import { EnvironmentFormProvider } from '../../features/robots/environment-form/provider';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const New = () => {
    return (
        <EnvironmentFormProvider>
            <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
                <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                    <Suspense fallback={<CenteredLoading />}>
                        <EnvironmentForm />
                    </Suspense>
                </View>
                <View gridArea='controls' backgroundColor={'gray-50'}>
                    <Preview />
                </View>
            </Grid>
        </EnvironmentFormProvider>
    );
};
