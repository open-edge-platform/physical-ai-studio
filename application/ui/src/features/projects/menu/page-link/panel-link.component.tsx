import { ReactNode } from 'react';

import { Flex, Icon, Link } from '@geti-ui/ui';
import { ChevronRightSmallLight } from '@geti-ui/ui/icons';

import classes from './panel-link.module.css';

type PanelLinkProps = {
    href: string;
    children: ReactNode;
};

export const PanelLink = ({ href, children }: PanelLinkProps) => (
    <Link href={href} UNSAFE_className={classes.panelLink}>
        <Flex alignItems='center' gap='size-100'>
            <Icon>
                <ChevronRightSmallLight />
            </Icon>
            {children}
        </Flex>
    </Link>
);
