import { Flex, Item, TabList, Tabs, View } from '@geti/ui';
import { useLocation } from 'react-router-dom';

import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';

const Header = ({ project_id }: { project_id: string }) => {
    return (
        <View backgroundColor={'gray-200'}>
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

export const TabNavigation = () => {
    const { project_id } = useProjectId();

    const { pathname } = useLocation();

    return (
        <Tabs
            aria-label='Header navigation'
            selectedKey={
                pathname.includes('cameras') ? 'cameras' : pathname.includes('environments') ? 'environments' : 'robots'
            }
            width='100%'
            gridArea='header'
        >
            <Header project_id={project_id} />
        </Tabs>
    );
};
