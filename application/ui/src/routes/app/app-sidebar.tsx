import { Item, TabList, View } from '@geti-ui/ui';

import { navItems } from './nav-items';

import classes from './app.layout.module.css';

export const AppSidebar = () => {
    return (
        <View gridArea='sidebar' backgroundColor={'gray-50'} height='100%' paddingTop={'size-500'}>
            <TabList
                height='100%'
                width='100%'
                aria-label='Main navigation items'
                UNSAFE_className={classes.sidebarList}
            >
                {navItems.map((item) => (
                    // Spread the href conditionally rather than passing `href={undefined}`: react-aria's
                    // useLinkProps treats an explicit `undefined` value as present (`'href' in props`),
                    // coercing it to an empty string on the rendered <div> for disabled items and
                    // triggering React's "empty string passed to href" warning.
                    <Item key={item.key} textValue={item.label} {...(item.enabled ? { href: item.path } : {})}>
                        {item.label}
                    </Item>
                ))}
            </TabList>
        </View>
    );
};
