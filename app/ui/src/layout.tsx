import { Flex, Grid, Item, TabList, Tabs, View } from '@geti/ui';
import { Outlet, useLocation, useNavigate } from 'react-router';

import { paths } from './router';

const Header = () => {
    const navigate = useNavigate();
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <View marginEnd='size-200' maxWidth={'5ch'}>
                    <span style={{ cursor: 'pointer' }} onClick={() => navigate(paths.projects.index.pattern)}>
                        Geti Action
                    </span>
                </View>
                <TabList
                    height={'100%'}
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item textValue='OpenAPI' key={'models'} href={paths.openapi({})}>
                        <Flex alignItems='center' gap='size-100'>
                            OpenAPI
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
                    <Outlet />
                </View>
            </Grid>
        </Tabs>
    );
};
