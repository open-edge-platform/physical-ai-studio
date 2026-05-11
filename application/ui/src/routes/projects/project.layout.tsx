import { Suspense } from 'react';

import { Divider } from '@adobe/react-spectrum';
import { ActionButton, DialogTrigger, Flex, Grid, Icon, Item, Link, Loading, TabList, Tabs, View } from '@geti-ui/ui';
import { Manifest } from '@geti-ui/ui/icons';
import { Outlet, useLocation } from 'react-router';

import { JobStatus } from '../../features/jobs/footer/job-status';
import { LogsDialog } from '../../features/logs/logs-dialog';
import { ProjectsListPanel } from '../../features/projects/menu/projects-list-panel.component';
import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';
import { ReactComponent as DatasetIcon } from './../../assets/icons/dataset-icon.svg';
import { ReactComponent as ModelsIcon } from './../../assets/icons/models-icon.svg';
import { ReactComponent as RobotIcon } from './../../assets/icons/robot-icon.svg';

const Header = ({ project_id }: { project_id: string }) => {
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <Link href='/' isQuiet variant='overBackground'>
                    <View marginEnd='size-200' maxWidth={'10ch'}>
                        <span style={{ whiteSpace: 'nowrap' }}>Physical AI</span> <span>Studio</span>
                    </View>
                </Link>

                <TabList
                    height={'100%'}
                    width={'100%'}
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item
                        textValue='Robot configuration'
                        key={'robots'}
                        href={paths.project.robots.index({ project_id })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <RobotIcon />
                            Robots
                        </Flex>
                    </Item>
                    <Item textValue='Datasets' key={'datasets'} href={paths.project.datasets.index({ project_id })}>
                        <Flex alignItems='center' gap='size-100'>
                            <DatasetIcon />
                            Datasets
                        </Flex>
                    </Item>
                    <Item textValue='Models' key={'models'} href={paths.project.models.index({ project_id })}>
                        <Flex alignItems='center' gap='size-100'>
                            <ModelsIcon />
                            Models
                        </Flex>
                    </Item>
                </TabList>
                <Flex alignItems={'center'} height={'100%'} marginStart='auto' gap='size-100'>
                    <ProjectsListPanel />
                </Flex>
            </Flex>
        </View>
    );
};

const getMainPageInProjectUrl = (pathname: string) => {
    const regexp = /\/projects\/[\w-]*\/([\w-]*)/g;
    const found = [...pathname.matchAll(regexp)];
    if (found.length) {
        const [, main] = found[0];
        if (main === 'cameras' || main === 'environments') {
            return 'robots';
        }
        return main;
    } else {
        return 'datasets';
    }
};

const Footer = () => {
    return (
        <View
            gridArea={'footer'}
            borderTopColor={'gray-75'}
            borderTopWidth={'thin'}
            borderBottomColor={'gray-75'}
            borderBottomWidth={'thin'}
            paddingX='size-100'
            paddingY='size-25'
        >
            <Flex alignItems={'center'} height='100%' gap='size-100'>
                <View>
                    <DialogTrigger type='fullscreen'>
                        <ActionButton
                            isQuiet
                            UNSAFE_style={{
                                paddingRight: 'var(--spectrum-global-dimension-size-100)',
                            }}
                        >
                            <Icon>
                                <Manifest />
                            </Icon>
                            Logs
                        </ActionButton>
                        {(close) => <LogsDialog close={close} />}
                    </DialogTrigger>
                </View>
                <Divider orientation='vertical' size='S' />
                <JobStatus />
            </Flex>
        </View>
    );
};

export const ProjectLayout = () => {
    const { project_id } = useProjectId();
    const { pathname } = useLocation();

    const pageName = getMainPageInProjectUrl(pathname);

    return (
        <Tabs aria-label='Header navigation' selectedKey={pageName} UNSAFE_style={{ height: '100%', minHeight: 0 }}>
            <Grid
                areas={['header', 'subheader', 'content', 'footer']}
                UNSAFE_style={{
                    gridTemplateRows:
                        // eslint-disable-next-line max-len
                        'var(--spectrum-global-dimension-size-800, 4rem) min-content auto var(--spectrum-global-dimension-size-400)',
                }}
                minHeight={0}
                height={'100%'}
            >
                <Header project_id={project_id} />
                <View gridArea={'content'} maxHeight={'100vh'} minHeight={0} height='100%'>
                    <Suspense fallback={<Loading mode='overlay' />}>
                        <Outlet />
                    </Suspense>
                </View>
                <Footer />
            </Grid>
        </Tabs>
    );
};
