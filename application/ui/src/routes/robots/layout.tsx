import { Suspense } from 'react';

import { Flex, Grid, Item, Loading, minmax, TabList, Tabs, View } from '@geti/ui';
import { Outlet, useLocation } from 'react-router-dom';

import { useProjectId } from '../../features/projects/use-project';
import { RobotsList } from '../../features/robots/robots-list';
import { paths } from '../../router';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

const Header = ({ project_id }: { project_id: string }) => {
    return (
        <View backgroundColor={'gray-200'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <TabList height={'100%'} width='100%'>
                    <Item
                        textValue='Robot configuration'
                        key={'robots'}
                        href={paths.project.robots.index({ project_id })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            Robot arms
                        </Flex>
                    </Item>
                    <Item textValue='Cameras' key={'cameras'} href={paths.project.cameras.index({ project_id })}>
                        <Flex alignItems='center' gap='size-100'>
                            Cameras
                        </Flex>
                    </Item>
                    <Item
                        textValue='Tele operation controller'
                        key={'teleoperators'}
                        href={paths.project.cameras.index({ project_id })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            Tele operation controller
                        </Flex>
                    </Item>
                    <Item
                        textValue='Environments'
                        key={'environments'}
                        href={paths.project.environments.index({ project_id })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            Environments
                        </Flex>
                    </Item>
                </TabList>
            </Flex>
        </View>
    );
};

export const RobotTabNavigation = () => {
    const { project_id } = useProjectId();

    const { pathname } = useLocation();

    return (
        <Tabs
            aria-label='Header navigation'
            selectedKey={
                pathname.includes('cameras') ? 'cameras' : pathname.includes('environments') ? 'environments' : 'robots'
            }
            width='100%'
            disabledKeys={['teleoperators', 'environments']}
        >
            <Header project_id={project_id} />
        </Tabs>
    );
};

export const Layout = () => {
    return (
        <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'} minHeight={0}>
            <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <RobotsList />
                </Suspense>
            </View>
            <View gridArea='controls' backgroundColor={'gray-50'} padding='size-400' minHeight={0}>
                <Suspense fallback={<CenteredLoading />}>
                    <Outlet />
                </Suspense>
            </View>
        </Grid>
    );
};
