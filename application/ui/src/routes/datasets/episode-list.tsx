import { Checkbox, Flex, View, VirtualizedListLayout } from '@geti/ui';
import { clsx } from 'clsx';

import { fetchClient } from '../../api/client';
import { EpisodeTag } from '../../features/datasets/episodes/episode-tag';
import { EpisodeSummary, useDataset } from './dataset-provider';

import classes from './episode-list.module.scss';

interface EpisodeListProps {
    episodes: EpisodeSummary[];
    onSelect: (index: number) => void;
    currentEpisode: number;
}

export const EpisodeList = ({ episodes, onSelect, currentEpisode }: EpisodeListProps) => {
    const { dataset_id, selectedEpisodes, setSelectedEpisodes } = useDataset();

    const toggleSelection = (episodeIndex: number) => {
        setSelectedEpisodes((list) =>
            list.includes(episodeIndex) ? list.filter((m) => m !== episodeIndex) : [...list, episodeIndex]
        );
    };

    return (
        <View UNSAFE_className={classes.episodePreviewList}>
            <VirtualizedListLayout
                items={episodes}
                ariaLabel='Episode list'
                containerHeight='100%'
                layoutOptions={{ rowHeight: 190 }}
                idFormatter={(episode) => `${episode.episode_index}`}
                textValueFormatter={(episode) => `Episode ${episode.episode_index + 1}`}
                renderItem={(episode) => {
                    const thumbnailSrc = fetchClient.PATH(
                        '/api/dataset/{dataset_id}/episodes/{episode_index}/thumbnail',
                        {
                            params: {
                                path: {
                                    dataset_id,
                                    episode_index: episode.episode_index,
                                },
                                query: { height: 240, width: 320 },
                            },
                        }
                    );

                    return (
                        <View
                            UNSAFE_className={clsx({
                                [classes.episodeItem]: true,
                                [classes.active]: currentEpisode === episode.episode_index,
                            })}
                        >
                            <img
                                alt={`Camera frame of ${episode.episode_index}`}
                                src={thumbnailSrc}
                                className={classes.episodeImage}
                                onClick={() => onSelect(episode.episode_index)}
                            />
                            <Flex alignItems={'center'} justifyContent={'space-between'} height='size-400' width='100%'>
                                <EpisodeTag episode={episode} variant='small' />
                                <Checkbox
                                    isSelected={selectedEpisodes.includes(episode.episode_index)}
                                    onPress={() => toggleSelection(episode.episode_index)}
                                    UNSAFE_className={classes.episodeCheckbox}
                                />
                            </Flex>
                        </View>
                    );
                }}
            />
        </View>
    );
};
