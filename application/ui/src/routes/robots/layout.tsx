import { Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti/ui';
import { Outlet } from 'react-router-dom';

import { RobotsList } from '../../features/robots/robots-list';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const Layout = () => {
    return (
        <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <RobotsList />
                </Suspense>
            </View>
            <View gridArea='controls' backgroundColor={'gray-50'} padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <Outlet />
                </Suspense>
            </View>
        </Grid>
    );
};
