import { Grid, minmax, View } from '@geti/ui';
import { Outlet } from 'react-router-dom';

import { RobotsList } from '../../features/robots/robots-list';

export const Layout = () => {
    return (
        <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                <RobotsList />
            </View>
            <View gridArea='controls' backgroundColor={'gray-50'} padding='size-400'>
                <Outlet />
            </View>
        </Grid>
    );
};
