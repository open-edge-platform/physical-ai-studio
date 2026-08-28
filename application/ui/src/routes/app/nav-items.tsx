import { ReactNode } from 'react';

import { Gear, ProjectsIcon } from '@geti-ui/ui/icons';

export type NavItem = {
    key: string;
    label: string;
    path: string;
    enabled: boolean;
    icon?: ReactNode;
};

export const navItems: NavItem[] = [
    { key: 'projects', label: 'Projects', path: '/projects', enabled: true, icon: <ProjectsIcon /> },
    { key: 'environments', label: 'Environments', path: '/environments', enabled: false },
    { key: 'settings', label: 'Settings', path: '/settings', enabled: true, icon: <Gear /> },
    { key: 'plugins', label: 'Plugins', path: '/plugins', enabled: false },
];

export const disabledNavItemKeys = navItems.filter((item) => !item.enabled).map((item) => item.key);
