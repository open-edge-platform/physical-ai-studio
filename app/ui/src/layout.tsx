import { Flex, Grid, Item, TabList, TabPanels, Tabs, View } from '@geti/ui';
import { Outlet, useLocation } from 'react-router';

import { paths } from './router';

const Header = () => {
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <View marginEnd='size-200' maxWidth={'5ch'}>
                    <span>Geti Action</span>
                </View>

                <TabList
                    height={'100%'}
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item textValue='Projects' key={paths.projects.index({})} href={paths.projects.index({})}>
                        <Flex alignItems='center' gap='size-100'>
                            Projects
                        </Flex>
                    </Item>
                    <Item
                        textValue='Life inference'
                        key={paths.robotConfiguration.index({})}
                        href={paths.robotConfiguration.index({})}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            Robot Configuration
                        </Flex>
                    </Item>
                    <Item textValue='Life inference' key={paths.datasets.index({})} href={paths.datasets.index({})}>
                        <Flex alignItems='center' gap='size-100'>
                            Datasets
                        </Flex>
                    </Item>
                    <Item textValue='Life inference' key={paths.models.index({})} href={paths.models.index({})}>
                        <Flex alignItems='center' gap='size-100'>
                            Models
                        </Flex>
                    </Item>
                </TabList>
            </Flex>
        </View>
    );
};

const getFirstPathSegment = (path: string): string => {
    const segments = path.split('/');
    return segments.length > 1 ? `/${segments[1]}` : '/';
};

export const Layout = () => {
    const { pathname } = useLocation();

    return (
        <Tabs aria-label='Header navigation' selectedKey={getFirstPathSegment(pathname)}>
            <Grid
                areas={['header', 'content']}
                UNSAFE_style={{
                    gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
                }}
                minHeight={'100vh'}
                maxHeight={'100vh'}
                height={'100%'}
            >
                <Header />
                <View backgroundColor={'gray-75'} gridArea={'content'}>
                    <TabPanels height={'100%'} UNSAFE_style={{ border: 'none' }}>
                        <Item textValue='index' key={paths.projects.index({})}>
                            <Outlet />
                        </Item>
                        <Item textValue='index' key={paths.robotConfiguration.index({})}>
                            <Outlet />
                        </Item>
                        <Item textValue='index' key={paths.datasets.index({})}>
                            <Outlet />
                        </Item>
                        <Item textValue='index' key={paths.models.index({})}>
                            <Outlet />
                        </Item>
                    </TabPanels>
                </View>
            </Grid>
        </Tabs>
    );
};
