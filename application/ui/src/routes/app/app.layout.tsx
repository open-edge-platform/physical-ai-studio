import { Suspense } from 'react';

import { Grid, Loading, Tabs, View } from '@geti-ui/ui';
import { Outlet, useLocation } from 'react-router';

import { AppSidebar } from './app-sidebar';
import { disabledNavItemKeys } from './nav-items';

import classes from './app.layout.module.css';

const getSelectedNavKey = (pathname: string) => {
    const [, firstSegment] = pathname.split('/');

    return firstSegment || 'projects';
};

export const AppLayout = () => {
    const { pathname } = useLocation();
    const selectedKey = getSelectedNavKey(pathname);

    return (
        <Tabs
            orientation='vertical'
            aria-label='Main navigation'
            selectedKey={selectedKey}
            disabledKeys={disabledNavItemKeys}
            UNSAFE_className={classes.layout}
            minHeight={0}
            height={'100%'}
            width={'100%'}
        >
            <Grid
                areas={['sidebar content']}
                rows={['minmax(0, 1fr)']}
                columns={['size-3000', 'minmax(0, 1fr)']}
                minHeight={0}
                height='100%'
                width={'100%'}
            >
                <AppSidebar />
                <View
                    gridArea='content'
                    minHeight={0}
                    height='100%'
                    backgroundColor={'gray-50'}
                    position={'relative'}
                    padding={'size-300'}
                >
                    <Suspense fallback={<Loading mode='overlay' />}>
                        <Outlet />
                    </Suspense>
                </View>
            </Grid>
        </Tabs>
    );
};
