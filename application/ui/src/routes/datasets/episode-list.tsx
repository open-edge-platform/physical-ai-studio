import { useMemo } from 'react';

import { Checkbox, Flex, Size, View, type Selection } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { fetchClient } from '../../api/client';
import { VirtualizedGridLayout } from '../../components/virtualized-grid-layout/virtualized-grid-layout';
import { EpisodeTag } from '../../features/datasets/episodes/episode-tag';
import { EpisodeThumbnail } from '../../features/datasets/episodes/episode-thumbnail/episode-thumbnail';
import { EpisodeSummary, useDataset } from './dataset-provider';

import classes from './episode-list.module.css';

interface EpisodeListProps {
    episodes: EpisodeSummary[];
    onSelect: (index: number) => void;
    currentEpisode: number | null;
    sortOrder: 'newest' | 'oldest';
    thumbnailCamera?: string;
    isThumbnailCameraLoading?: boolean;
}

const episodeGridLayoutOptions = {
    minItemSize: new Size(200, 180),
    maxItemSize: new Size(280, 252),
    preserveAspectRatio: true,
    minSpace: new Size(12, 12),
};

export const EpisodeList = ({
    episodes,
    onSelect,
    currentEpisode,
    sortOrder,
    thumbnailCamera,
    isThumbnailCameraLoading = false,
}: EpisodeListProps) => {
    const { dataset_id, selectedEpisodes, setSelectedEpisodes } = useDataset();
    const sortedEpisodes = useMemo(
        () =>
            [...episodes].sort((a, b) =>
                sortOrder === 'newest' ? b.episode_index - a.episode_index : a.episode_index - b.episode_index
            ),
        [episodes, sortOrder]
    );

    const toggleSelection = (episodeIndex: number) => {
        setSelectedEpisodes((list) =>
            list.includes(episodeIndex) ? list.filter((m) => m !== episodeIndex) : [...list, episodeIndex]
        );
    };

    const selectedKeys = currentEpisode !== null ? new Set([String(currentEpisode)]) : new Set<string>();

    const handleSelectionChange = (keys: Selection) => {
        if (keys === 'all') {
            return;
        }

        onSelect(Number(keys.values().next().value));
    };

    return (
        <View UNSAFE_className={classes.episodePreviewList}>
            <div className={classes.episodePreviewListInner}>
                <VirtualizedGridLayout
                    items={sortedEpisodes}
                    ariaLabel='Episode list'
                    containerHeight='100%'
                    layoutOptions={episodeGridLayoutOptions}
                    idFormatter={(episode) => String(episode.episode_index)}
                    textValueFormatter={(episode) => `Episode ${episode.episode_index + 1}`}
                    selectionMode='single'
                    disallowEmptySelection
                    selectedKeys={selectedKeys}
                    onSelectionChange={handleSelectionChange}
                    renderItem={(episode) => {
                        const thumbnailSrc =
                            thumbnailCamera === undefined
                                ? undefined
                                : fetchClient.PATH('/api/dataset/{dataset_id}/episodes/{episode_index}/thumbnail', {
                                      params: {
                                          path: { dataset_id, episode_index: episode.episode_index },
                                          query: { height: 240, width: 320, camera: thumbnailCamera },
                                      },
                                  });

                        const episodeVideo =
                            thumbnailCamera === undefined ? undefined : episode.videos[thumbnailCamera];

                        const previewVideo =
                            episodeVideo === undefined
                                ? undefined
                                : {
                                      url: fetchClient.PATH('/api/dataset/{dataset_id}/video/{video_path}', {
                                          params: { path: { dataset_id, video_path: episodeVideo.path } },
                                      }),
                                      start: episodeVideo.start,
                                      end: episodeVideo.end,
                                      fps: episode.fps,
                                  };

                        return (
                            <View
                                UNSAFE_className={clsx(classes.episodeItem, {
                                    [classes.active]: currentEpisode === episode.episode_index,
                                })}
                            >
                                <EpisodeThumbnail
                                    key={thumbnailSrc ?? 'pending'}
                                    alt={`Camera frame of ${episode.episode_index}`}
                                    src={thumbnailSrc}
                                    isLoading={isThumbnailCameraLoading}
                                    video={previewVideo}
                                />
                                <Flex
                                    alignItems={'center'}
                                    justifyContent={'space-between'}
                                    height='size-400'
                                    width='100%'
                                >
                                    <EpisodeTag episode={episode} variant='small' />

                                    <Checkbox
                                        isEmphasized
                                        isSelected={selectedEpisodes.includes(episode.episode_index)}
                                        onPress={() => toggleSelection(episode.episode_index)}
                                        UNSAFE_className={classes.episodeCheckbox}
                                    />
                                </Flex>
                            </View>
                        );
                    }}
                />
            </div>
        </View>
    );
};
