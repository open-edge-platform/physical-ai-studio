import { Item, TabList, View } from '@geti-ui/ui';

import { navItems } from './nav-items';

import classes from './app.layout.module.css';

export const AppSidebar = () => {
    return (
        <View
            gridArea='sidebar'
            backgroundColor={'gray-100'}
            height='100%'
            paddingTop={'size-500'}
            borderEndWidth={'thin'}
            borderEndColor={'gray-50'}
        >
            <TabList
                height='100%'
                width='100%'
                aria-label='Main navigation items'
                UNSAFE_className={classes.sidebarList}
            >
                {navItems
                    .filter((item) => item.enabled)
                    .map((item) => (
                        <Item key={item.key} textValue={item.label} href={item.path}>
                            {item.label}
                        </Item>
                    ))}
            </TabList>
        </View>
    );
};
