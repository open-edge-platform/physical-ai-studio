import { ActionButton, Flex, Grid, Item, TabList, Tabs, View } from '@geti/ui';
import { ChevronLeft } from '@geti/ui/icons';
import { Outlet, redirect, useLocation, useNavigate, useParams } from 'react-router';

import { paths } from '../../router';
import { ReactComponent as DatasetIcon } from './../../assets/icons/dataset-icon.svg';
import { ReactComponent as ModelsIcon } from './../../assets/icons/models-icon.svg';
import { ReactComponent as RobotIcon } from './../../assets/icons/robot-icon.svg';
import { ReactComponent as TestsIcon } from './../../assets/icons/tests-icon.svg';
import { ProjectProvider } from './project.provider';

const Header = ({ project_id }: { project_id: string }) => {
    const navigate = useNavigate();
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <View marginEnd='size-200' maxWidth={'5ch'}>
                    <span>Geti Action</span>
                </View>
                <ActionButton isQuiet onPress={() => navigate(paths.projects.index.pattern)}>
                    <ChevronLeft fill={'white'} />
                </ActionButton>

                <TabList
                    height={'100%'}
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item
                        textValue='Robot configuration'
                        key={'robots'}
                        href={paths.project.robotConfiguration({ project_id })}
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
                    <Item textValue='Models' key={'models'} href={paths.project.models({ project_id })}>
                        <Flex alignItems='center' gap='size-100'>
                            <ModelsIcon />
                            Models
                        </Flex>
                    </Item>
                    <Item textValue='Cameras' key={'cameras'} href={paths.project.cameras.index({ project_id })}>
                        <Flex alignItems='center' gap='size-100'>
                            Cameras
                        </Flex>
                    </Item>
                    <Item textValue='OpenAPI' key={'openapi'} href={paths.openapi({})}>
                        <Flex alignItems='center' gap='size-100'>
                            <TestsIcon />
                            OpenAPI
                        </Flex>
                    </Item>
                </TabList>
            </Flex>
        </View>
    );
};

const getMainPageInProjectUrl = (pathname: string) => {
    const regexp = /\/projects\/[\w-]*\/([\w-]*)/g;
    const found = [...pathname.matchAll(regexp)];
    if (found.length) {
        const [_base, main] = found[0];
        return main;
    } else {
        return 'datasets';
    }
};

export const ProjectLayout = () => {
    const { project_id } = useParams();
    const { pathname } = useLocation();

    const pageName = getMainPageInProjectUrl(pathname);

    if (project_id === undefined) {
        redirect(paths.root.pattern);
    } else {
        return (
            <ProjectProvider project_id={project_id}>
                <Tabs aria-label='Header navigation' selectedKey={pageName}>
                    <Grid
                        areas={['header', 'content']}
                        UNSAFE_style={{
                            gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
                        }}
                        minHeight={'100vh'}
                        maxHeight={'100vh'}
                        height={'100%'}
                    >
                        <Header project_id={project_id} />
                        <View backgroundColor={'gray-75'} gridArea={'content'}>
                            <Outlet />
                        </View>
                    </Grid>
                </Tabs>
            </ProjectProvider>
        );
    }
};
