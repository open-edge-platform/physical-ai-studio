import { Flex, Item, TabList, Text, View } from '@geti-ui/ui';

import { AppLogo } from '../../components/app-logo/app-logo';
import { navItems } from './nav-items';

import classes from './app.layout.module.css';

export const AppSidebar = () => {
    return (
        <View
            gridArea='sidebar'
            backgroundColor={'gray-100'}
            height='100%'
            borderEndWidth={'thin'}
            borderEndColor={'gray-50'}
        >
            <Flex direction='column' height='100%' minHeight={0}>
                <View height='size-800'>
                    <Flex height='100%' alignItems={'center'} marginX='1rem'>
                        <AppLogo />
                    </Flex>
                </View>
                <View flex={1} minHeight={0} paddingTop={'size-500'}>
                    <TabList width='100%' aria-label='Main navigation items' UNSAFE_className={classes.sidebarList}>
                        {navItems
                            .filter((item) => item.enabled)
                            .map((item) => (
                                <Item key={item.key} textValue={item.label} href={item.path}>
                                    <Flex alignItems={'center'} gap={'size-100'}>
                                        <Flex UNSAFE_className={classes.sidebarListItemIcon}>{item.icon}</Flex>
                                        <Text>{item.label}</Text>
                                    </Flex>
                                </Item>
                            ))}
                    </TabList>
                </View>
            </Flex>
        </View>
    );
};
