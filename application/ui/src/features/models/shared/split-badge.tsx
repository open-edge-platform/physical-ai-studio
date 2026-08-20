import { Badge } from '@adobe/react-spectrum';
import { Flex } from '@geti-ui/ui';

import classes from './split-badge.module.css';

interface SplitBadgeProps {
    first: string;
    second: string;
}

export const SplitBadge = ({ first, second }: SplitBadgeProps) => {
    return (
        <Flex UNSAFE_className={classes.badgeWrapper}>
            <Badge variant={'positive'} UNSAFE_className={classes.badgeLeft}>
                {first}
            </Badge>
            <Badge variant={'info'} UNSAFE_className={classes.badgeRight}>
                {second}
            </Badge>
        </Flex>
    );
};

interface SingleBadgeProps {
    text: string;
    color: string;
    title?: string;
    /** Preserve the exact casing of `text` instead of capitalizing each word. */
    preserveCase?: boolean;
}

export const SingleBadge = ({ text, color, title, preserveCase }: SingleBadgeProps) => {
    return (
        <span title={title ?? text} className={classes.badgeWrapper}>
            <Badge
                variant={'info'}
                UNSAFE_className={classes.badge}
                UNSAFE_style={{ backgroundColor: color, ...(preserveCase && { textTransform: 'none' }) }}
            >
                {text}
            </Badge>
        </span>
    );
};
