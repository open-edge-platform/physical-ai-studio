import { Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti-ui/ui';

import { CreateRobotForm } from '../../features/robots/robot-form/create-form';
import { Preview } from '../../features/robots/robot-form/preview';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const New = () => {
    return (
        <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <CreateRobotForm />
                </Suspense>
            </View>
            <View gridArea='controls'>
                <Preview />
            </View>
        </Grid>
    );
};
