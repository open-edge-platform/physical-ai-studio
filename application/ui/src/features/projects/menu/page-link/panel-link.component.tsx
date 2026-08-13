import { ReactNode } from 'react';

import { Flex, Icon, Link } from '@geti-ui/ui';
import { ChevronRightSmallLight } from '@geti-ui/ui/icons';

import styles from './panel-link.module.css';

type PanelLinkProps = {
    href: string;
    children: ReactNode;
};

export const PanelLink = ({ href, children }: PanelLinkProps) => (
    <Link href={href} UNSAFE_className={styles.panelLink}>
        <Flex alignItems='center' gap='size-100'>
            <Icon>
                <ChevronRightSmallLight />
            </Icon>
            {children}
        </Flex>
    </Link>
);
