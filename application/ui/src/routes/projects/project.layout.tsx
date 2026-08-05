import { Suspense } from 'react';

import {
    ActionButton,
    DialogTrigger,
    Divider,
    Flex,
    Grid,
    Icon,
    Item,
    Loading,
    TabList,
    Tabs,
    View,
} from '@geti-ui/ui';
import { Manifest } from '@geti-ui/ui/icons';
import { Outlet, useLocation } from 'react-router';

import { AppLogo } from '../../components/app-logo/app-logo';
import { JobStatus } from '../../features/jobs/footer/job-status';
import { LogsDialog } from '../../features/logs/logs-dialog';
import { ProjectsListPanel } from '../../features/projects/menu/projects-list-panel.component';
import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';
import { getMainPageInProjectUrl } from './project-navigation';

const Header = ({ project_id }: { project_id: string }) => {
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <AppLogo />

                <TabList
                    height={'100%'}
                    width={'100%'}
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
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
                    <ProjectsListPanel />
                </Flex>
            </Flex>
        </View>
    );
};

const Footer = () => {
    return (
        <View
            gridArea={'footer'}
            borderTopColor={'gray-300'}
            borderTopWidth={'thin'}
            borderBottomColor={'gray-75'}
            borderBottomWidth={'thin'}
            paddingX='size-100'
            paddingY='size-25'
        >
            <Flex alignItems={'center'} height='100%' gap='size-100'>
                <View overflow={'hidden'}>
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
                    gridTemplateColumns: 'minmax(0, 1fr)',
                    gridTemplateRows:
                        // eslint-disable-next-line max-len
                        'var(--spectrum-global-dimension-size-800, 4rem) min-content minmax(0, 1fr) var(--spectrum-global-dimension-size-400)',
                }}
                minHeight={0}
                height={'100%'}
            >
                <Header project_id={project_id} />
                <View gridArea={'content'} maxHeight={'100vh'} minWidth={0} minHeight={0} height='100%'>
                    <Suspense fallback={<Loading mode='overlay' />}>
                        <Outlet />
                    </Suspense>
                </View>
                <Footer />
            </Grid>
        </Tabs>
    );
};
