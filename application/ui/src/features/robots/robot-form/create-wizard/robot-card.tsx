import { ReactNode, useState } from 'react';

import { DimensionValue, Flex, Text } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { RobotThumbnail } from './robot-thumbnail';

import classes from './robot-card.module.css';

export const CardTitle = ({ children }: { children: ReactNode }) => (
    <Text UNSAFE_className={classes.title}>{children}</Text>
);

export const CardMeta = ({ children }: { children: ReactNode }) => (
    <Text UNSAFE_className={classes.meta}>{children}</Text>
);

interface RobotCardProps {
    /** Robot type whose URDF is shown in the thumbnail. */
    type: string;
    thumbnailHeight: DimensionValue;
    isSelected: boolean;
    onSelect: () => void;
    children: ReactNode;
}

export const RobotCard = ({ type, thumbnailHeight, isSelected, onSelect, children }: RobotCardProps) => {
    const [isHovered, setIsHovered] = useState(false);

    return (
        <button
            type='button'
            aria-pressed={isSelected}
            className={clsx(classes.card, { [classes.cardSelected]: isSelected })}
            onClick={onSelect}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onFocus={() => setIsHovered(true)}
            onBlur={() => setIsHovered(false)}
        >
            <Flex direction='column' gap='size-100'>
                <RobotThumbnail type={type} isActive={isHovered} height={thumbnailHeight} />
                {children}
            </Flex>
        </button>
    );
};
