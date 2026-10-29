import { ReactNode } from 'react';

import { Flex, Text } from '@geti-ui/ui';

import classes from './recording-sidebar.module.css';

interface SidebarSectionProps {
    label?: string;
    children: ReactNode;
}

export const SidebarSection = ({ label, children }: SidebarSectionProps) => (
    <Flex direction='column' gap='size-150' minWidth={0}>
        {label !== undefined && <Text UNSAFE_className={classes.sectionLabel}>{label}</Text>}
        {children}
    </Flex>
);
