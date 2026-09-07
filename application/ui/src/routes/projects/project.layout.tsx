import { Suspense } from 'react';

import { Flex, Grid, Item, Loading, TabList, Tabs, View } from '@geti-ui/ui';
import { Outlet, useLocation } from 'react-router';

import { AppFooter } from '../../components/app-footer/app-footer';
import { AppLogo } from '../../components/app-logo/app-logo';
import { ProjectMenu } from '../../features/projects/menu/project-menu.component';
import { useProject, useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';
import { getMainPageInProjectUrl } from './project-navigation';

const Header = ({ project_id }: { project_id: string }) => {
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <AppLogo />

                <TabList height={'100%'} width={'100%'}>
                    {[
                        <Item
                            textValue='Robot configuration'
                            key={'robots'}
                            href={paths.project.robots.index({ project_id })}
                        >
                            <Flex alignItems='center' gap='size-100'>
                                Robots
                            </Flex>
                        </Item>,
                        <Item textValue='Datasets' key={'datasets'} href={paths.project.datasets.index({ project_id })}>
                            <Flex alignItems='center' gap='size-100'>
                                Datasets
                            </Flex>
                        </Item>,
                        <Item textValue='Models' key={'models'} href={paths.project.models.index({ project_id })}>
                            <Flex alignItems='center' gap='size-100'>
                                Models
                            </Flex>
                        </Item>,
                    ]}
                </TabList>
                <Flex alignItems={'center'} height={'100%'} marginStart='auto' gap='size-100'>
                    <ProjectMenu />
                </Flex>
            </Flex>
        </View>
    );
};

export const ProjectLayout = () => {
    const { project_id } = useProjectId();
    const { pathname } = useLocation();

    // We want to check if the project exists before rendering the layout. If it doesn't, error boundary will catch it.
    useProject();

    const pageName = getMainPageInProjectUrl(pathname);

    return (
        <Tabs aria-label='Header navigation' selectedKey={pageName} UNSAFE_style={{ height: '100%', minHeight: 0 }}>
            <Grid
                areas={['header', 'subheader', 'content', 'footer']}
                UNSAFE_style={{
                    gridTemplateColumns: 'minmax(0, 1fr)',
                    gridTemplateRows:
                        // eslint-disable-next-line max-len
                        'var(--spectrum-global-dimension-size-800, 4rem) min-content minmax(0, 1fr) var(--spectrum-global-dimension-size-400)',
                }}
                minHeight={0}
                height={'100%'}
            >
                <Header project_id={project_id} />
                <View
                    gridArea={'content'}
                    maxHeight={'100vh'}
                    minWidth={0}
                    minHeight={0}
                    height='100%'
                    backgroundColor={'gray-75'}
                >
                    <Suspense fallback={<Loading mode='overlay' />}>
                        <Outlet />
                    </Suspense>
                </View>
                <AppFooter />
            </Grid>
        </Tabs>
    );
};
