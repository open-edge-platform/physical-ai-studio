import { Divider, Flex, Item, ListView, Selection, Tag, Text, View } from '@geti/ui';
import { SchemaEpisode } from "../../../api/openapi-spec";

import classes from './episode-tag.module.scss';
import { toMMSS } from '../../../utils';
import clsx from 'clsx';

interface EpisodeTagProps {
    episode: SchemaEpisode;
    variant: 'small' | 'medium';
}

export const EpisodeTag = ({ episode, variant }: EpisodeTagProps) => {
    return (
        <Flex gap="size-100">
            <div className={clsx(classes.episodeIndex, { [classes.variantSmall]: variant === 'small' })}>E{episode.episode_index + 1}</div>
            <div className={clsx(classes.episodeDuration, { [classes.variantSmall]: variant === 'small' })}>{toMMSS(episode.length / episode.fps)}</div>
        </Flex>
    )
}
