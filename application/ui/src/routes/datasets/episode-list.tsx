import { Checkbox, Flex, View } from '@geti/ui';
import { clsx } from 'clsx';

import { SchemaEpisode } from '../../api/openapi-spec';
import { EpisodeTag } from '../../features/datasets/episodes/episode-tag';
import { useDataset } from './dataset-provider';

import classes from './episode-list.module.scss';

interface EpisodeListProps {
    episodes: SchemaEpisode[];
    onSelect: (index: number) => void;
    currentEpisode: number;
}

export const EpisodeList = ({ episodes, onSelect, currentEpisode }: EpisodeListProps) => {
    const { selectedEpisodes, setSelectedEpisodes } = useDataset();

    const toggleSelection = (episodeIndex: number) => {
        setSelectedEpisodes((list) =>
            list.includes(episodeIndex) ? list.filter((m) => m !== episodeIndex) : [...list, episodeIndex]
        );
    };

    return (
        <Flex flex height='100%' direction='column' maxWidth={256}>
            <Flex flex={'1 1 0'} direction='column' minHeight={0}>
                <View UNSAFE_className={classes.episodePreviewList}>
                    {episodes.map((episode) => (
                        <View
                            UNSAFE_className={clsx({
                                [classes.episodeItem]: true,
                                [classes.active]: currentEpisode === episode.episode_index,
                            })}
                            key={episode.episode_index}
                        >
                            <img
                                alt={`Camera frame of ${episode.episode_index}`}
                                src={`data:image/jpg;base64,${episode.thumbnail}`}
                                style={{
                                    objectFit: 'contain',
                                    width: '100%',
                                    height: '100%',
                                }}
                                onClick={() => onSelect(episode.episode_index)}
                            />
                            <EpisodeTag episode={episode} variant='small' />
                            <Checkbox
                                isSelected={selectedEpisodes.includes(episode.episode_index)}
                                onPress={() => toggleSelection(episode.episode_index)}
                                UNSAFE_className={classes.episodeCheckbox}
                            />
                        </View>
                    ))}
                </View>
            </Flex>
        </Flex>
    );
};
