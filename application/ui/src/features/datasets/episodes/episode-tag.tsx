import { Divider, Flex, Item, ListView, Selection, Tag, Text, View } from '@geti/ui';
import { SchemaEpisode } from "../../../api/openapi-spec";

import classes from './episode-tag.module.scss';
import { toMMSS } from '../../../utils';

export const EpisodeTag = ({ episode }: { episode: SchemaEpisode }) => {
    return (
        <Flex gap="size-100">
            <div className={classes.episodeIndex}>E{episode.episode_index + 1}</div>
          <div className={classes.episodeDuration}>{toMMSS(episode.length / episode.fps)}</div>
        </Flex>
    )
}
