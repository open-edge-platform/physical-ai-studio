import { Badge } from "@adobe/react-spectrum";
import { Flex } from "@geti/ui"

import classes from './split-badge.module.scss';

interface SplitBadgeProps {
    first: string;
    second: string;
}

export const SplitBadge = ({ first, second }: SplitBadgeProps) => {
    return (
        <Flex>
            <Badge variant={'positive'} UNSAFE_className={classes.badgeLeft}>{first}</Badge>
            <Badge variant={'info'} UNSAFE_className={classes.badgeRight}>{second}</Badge>
        </Flex>
    );
}
