import { Flex } from '@geti/ui';
import { clsx } from 'clsx';

import { SchemaEpisode } from '../../../api/openapi-spec';
import { toMMSS } from '../../../utils';

import classes from './episode-tag.module.scss';

interface EpisodeTagProps {
    episode: SchemaEpisode;
    variant: 'small' | 'medium';
}

export const EpisodeTag = ({ episode, variant }: EpisodeTagProps) => {
    return (
        <Flex gap='size-100'>
            <div className={clsx(classes.episodeIndex, { [classes.variantSmall]: variant === 'small' })}>
                E{episode.episode_index + 1}
            </div>
            <div className={clsx(classes.episodeDuration, { [classes.variantSmall]: variant === 'small' })}>
                {toMMSS(episode.length / episode.fps)}
            </div>
        </Flex>
    );
};
