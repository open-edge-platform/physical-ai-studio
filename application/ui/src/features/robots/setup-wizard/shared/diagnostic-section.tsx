import type { ReactNode } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, Text } from '@geti/ui';

import { StatusBadge, type StatusBadgeVariant } from './status-badge';

import classes from './setup-wizard.module.scss';

interface DiagnosticSectionBadge {
    label: string;
    variant: StatusBadgeVariant;
}

interface DiagnosticSectionProps {
    title: string;
    badge: DiagnosticSectionBadge;
    defaultExpanded?: boolean;
    children: ReactNode;
}

export const DiagnosticSection = ({ title, badge, defaultExpanded, children }: DiagnosticSectionProps) => (
    <Disclosure defaultExpanded={defaultExpanded} isQuiet>
        <DisclosureTitle UNSAFE_className={classes.disclosureHeader}>
            <Flex alignItems='center' gap='size-100' width='100%'>
                <Text UNSAFE_style={{ fontWeight: 600, fontSize: 14 }}>{title}</Text>
                <Flex flex alignItems='center' justifyContent='end'>
                    <StatusBadge variant={badge.variant}>{badge.label}</StatusBadge>
                </Flex>
            </Flex>
        </DisclosureTitle>
        <DisclosurePanel>{children}</DisclosurePanel>
    </Disclosure>
);
