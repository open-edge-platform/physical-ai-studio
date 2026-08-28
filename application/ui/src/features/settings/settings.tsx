import { ReactNode, Suspense } from 'react';

import { Item, Loading, TabList, TabPanels, Tabs } from '@geti-ui/ui';
import { useMatch } from 'react-router';

import { paths } from '../../router';
import { Compute } from './compute';
import { GeneralSettings } from './general/general-settings';

type TabItem = {
    key: string;
    name: string;
    href: string;
    content: ReactNode;
};

const useActiveTab = () => {
    const match = useMatch(paths.settings.index.path(':activeTab').pattern);

    return match?.params?.activeTab ?? 'general';
};

export const SettingsView = () => {
    const activeTab = useActiveTab();

    const tabs: TabItem[] = [
        {
            key: 'general',
            name: 'General',
            href: paths.settings.index.pattern,
            content: (
                <Suspense fallback={<Loading />}>
                    <GeneralSettings />
                </Suspense>
            ),
        },
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
        <Tabs items={tabs} selectedKey={activeTab} height={'100%'} minHeight={0}>
            <TabList aria-label={'Settings tabs'}>
                {(tabItem: TabItem) => (
                    <Item key={tabItem.key} href={tabItem.href}>
                        {tabItem.name}
                    </Item>
                )}
            </TabList>
            <TabPanels minHeight={0} UNSAFE_style={{ overflowY: 'auto', scrollbarGutter: 'stable' }}>
                {(tabItem: TabItem) => <Item key={tabItem.key}>{tabItem.content}</Item>}
            </TabPanels>
        </Tabs>
    );
};
