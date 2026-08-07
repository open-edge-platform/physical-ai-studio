export interface NavItem {
    key: string;
    label: string;
    path: string;
    enabled: boolean;
}

export const navItems: NavItem[] = [
    { key: 'projects', label: 'Projects', path: '/projects', enabled: true },
    { key: 'environments', label: 'Environments', path: '/environments', enabled: false },
    { key: 'settings', label: 'Settings', path: '/settings', enabled: true },
    { key: 'plugins', label: 'Plugins', path: '/plugins', enabled: false },
];

export const disabledNavItemKeys = navItems.filter((item) => !item.enabled).map((item) => item.key);
