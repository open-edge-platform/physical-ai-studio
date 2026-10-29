import { ReactNode } from 'react';

import { Badge, Flex } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { EpisodeSummary } from '../../../routes/datasets/dataset-provider';
import { toMMSS } from '../../../utils';

import classes from './episode-tag.module.css';

interface EpisodeTagProps {
    indicator: number;
    variant?: 'small' | 'medium';
    children?: ReactNode;
}

export const EpisodeTag = ({ indicator, variant = 'medium', children }: EpisodeTagProps) => {
    return (
        <Flex gap='size-100' alignItems='center'>
            <Badge variant='yellow' UNSAFE_className={classes.badge}>{`E${indicator}`}</Badge>
            {children != null && children !== false && (
                <Flex
                    alignItems='center'
                    UNSAFE_className={clsx(classes.label, { [classes.variantSmall]: variant === 'small' })}
                >
                    {children}
                </Flex>
            )}
        </Flex>
    );
};

interface EpisodeDurationTagProps {
    episode: EpisodeSummary;
    variant?: 'small' | 'medium';
}

export const EpisodeDurationTag = ({ episode, variant }: EpisodeDurationTagProps) => {
    return (
        <EpisodeTag indicator={episode.episode_index + 1} variant={variant}>
            {toMMSS(episode.length / episode.fps)}
        </EpisodeTag>
    );
};
