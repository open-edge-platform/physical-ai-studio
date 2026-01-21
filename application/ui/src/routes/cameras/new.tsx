import { Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti/ui';

import { CameraForm } from '../../features/robots/camera-form/form';
import { Preview } from '../../features/robots/camera-form/preview';
import { CameraFormProvider } from '../../features/robots/camera-form/provider';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const New = () => {
    return (
        <CameraFormProvider>
            <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
                <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                    <Suspense fallback={<CenteredLoading />}>
                        <CameraForm />
                    </Suspense>
                </View>
                <View gridArea='controls' backgroundColor={'gray-50'} padding='size-400'>
                    <Preview />
                </View>
            </Grid>
        </CameraFormProvider>
    );
};
