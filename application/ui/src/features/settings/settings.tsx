import { ReactNode } from 'react';

import { Item, TabList, TabPanels, Tabs } from '@geti-ui/ui';
import { useMatch } from 'react-router';

import { paths } from '../../router';
import { Compute } from './compute';

type TabItem = {
    key: string;
    name: string;
    href: string;
    content: ReactNode;
};

const useActiveTab = () => {
    const match = useMatch(paths.settings.index.path(':activeTab').pattern);

    return match?.params?.activeTab ?? 'compute';
};

export const SettingsView = () => {
    const activeTab = useActiveTab();

    const tabs: TabItem[] = [
        /*{
            key: 'general',
            name: 'General',
            href: paths.settings.index.pattern,
            content: <></>,
        },*/
        {
            key: 'compute',
            name: 'Compute',
            href: paths.settings.compute.pattern,
            content: <Compute />,
        },
        /*{
            key: 'storage',
            name: 'Storage',
            href: paths.settings.storage.pattern,
            content: <></>,
        },
        {
            key: 'about',
            name: 'About',
            href: paths.settings.about.pattern,
            content: <></>,
        },*/
    ];

    return (
        <Tabs items={tabs} selectedKey={activeTab}>
            <TabList aria-label={'Settings tabs'}>
                {(tabItem: TabItem) => (
                    <Item key={tabItem.key} href={tabItem.href}>
                        {tabItem.name}
                    </Item>
                )}
            </TabList>
            <TabPanels>{(tabItem: TabItem) => <Item key={tabItem.key}>{tabItem.content}</Item>}</TabPanels>
        </Tabs>
    );
};
