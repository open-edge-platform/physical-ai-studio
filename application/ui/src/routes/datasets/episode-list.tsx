import { SchemaEpisode } from "../../api/openapi-spec"
import { Divider, Flex, Item, ListView, Selection, Text, View } from '@geti/ui';

import classes from './episode-list.module.scss';
import clsx from "clsx";

interface EpisodeListProps {
    episodes: SchemaEpisode[];
    onSelect: (index: number) => void;
    currentEpisode: number;
}

export const EpisodeList = ({ episodes, onSelect, currentEpisode }: EpisodeListProps) => {
    return (
        <Flex flex height="100%" direction="column" maxWidth={256} >
            <Flex flex={"1 1 0"} direction='column' minHeight={0}>
                <View UNSAFE_style={{ overflowY: 'scroll', maxHeight: "100%" }} UNSAFE_className={classes.episodePreviewList}>
                    {episodes.map((episode) => (
                        <View
                            UNSAFE_className={clsx({ [classes.episodeItem]: true, [classes.active]: currentEpisode === episode.episode_index })}
                            key={episode.episode_index}>
                            <img
                                alt={`Camera frame of ${episode.episode_index}`}
                                src={`data:image/jpg;base64,${episode.thumbnail}`}
                                style={{
                                    objectFit: 'contain',
                                    width: "100%",
                                    height: "100%",
                                }}
                                onClick={() => onSelect(episode.episode_index)}
                            />
                        </View>
                    ))}
                </View>
            </Flex>
        </Flex>

    );
}
