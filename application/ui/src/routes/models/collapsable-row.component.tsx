import { ReactNode, useState } from 'react';

import { Flex, View } from '@geti-ui/ui';
import { ChevronDownSmallLight, ChevronRightSmallLight } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';

import classes from './collapsable-row.module.scss';

interface CollapsableRowProps {
    header: ReactNode;
    children: ReactNode;
}

export const CollapsableRow = ({ header, children }: CollapsableRowProps) => {
    const [collapsed, setCollapsed] = useState<boolean>(true);

    return (
        <View>
            <div
                onClick={() => setCollapsed((m) => !m)}
                className={clsx({
                    [classes.collapsableRow]: true,
                    [classes.collapsableRowCollapsed]: collapsed,
                })}
            >
                <Flex alignItems='center'>
                    <Flex UNSAFE_className={classes.collapseButton}>
                        {collapsed ? <ChevronDownSmallLight fill='white' /> : <ChevronRightSmallLight fill='white' />}
                    </Flex>
                    {header}
                </Flex>
            </div>
            {!collapsed && children}
        </View>
    );
};
