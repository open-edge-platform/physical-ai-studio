import { Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti-ui/ui';
import { Outlet } from 'react-router';

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
        <Grid
            areas={['robot controls']}
            columns={['size-6000', minmax(0, '1fr')]}
            rows={[minmax(0, '1fr')]}
            height={'100%'}
            minHeight={0}
        >
            <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <RobotsList />
                </Suspense>
            </View>
            <View gridArea='controls' backgroundColor={'gray-75'} minHeight={0} minWidth={0} overflow='auto'>
                <Suspense fallback={<CenteredLoading />}>
                    <Outlet />
                </Suspense>
            </View>
        </Grid>
    );
};
