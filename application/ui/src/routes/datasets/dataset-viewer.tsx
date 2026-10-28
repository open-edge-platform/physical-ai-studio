import { useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    Content,
    DialogTrigger,
    Flex,
    Heading,
    IllustratedMessage,
    Loading,
    Text,
    View,
} from '@geti-ui/ui';
import { Add, Delete, SortUpDown } from '@geti-ui/ui/icons';
import { keepPreviousData } from '@tanstack/react-query';
import { Group, Panel, Separator, useDefaultLayout } from 'react-resizable-panels';

import { $api } from '../../api/client';
import { useDeleteEpisodeQuery } from '../../features/datasets/episodes/use-episodes';
import { RecordDatasetButton } from '../../features/datasets/recording-environments/record-dataset-button';
import { ThumbnailCameraMenu } from '../../features/datasets/thumbnail-camera/thumbnail-camera-menu';
import { useThumbnailCamera } from '../../features/datasets/thumbnail-camera/use-thumbnail-camera';
import { useProjectId } from '../../features/projects/use-project';
import { pluralize } from '../../utils';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { useDataset } from './dataset-provider';
import { EpisodeList } from './episode-list';
import { EpisodeViewer } from './episode-viewer';
import { useActiveEpisode } from './use-active-episode';

import classes from './dataset-viewer.module.css';

export const DatasetViewer = () => {
    const { dataset, episodes, selectedEpisodes, setSelectedEpisodes } = useDataset();
    const { project_id } = useProjectId();

    const { deleteEpisodes, isPending } = useDeleteEpisodeQuery(dataset.id!);
    const [activeEpisodeIndex, setActiveEpisodeIndex] = useActiveEpisode();
    const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');
    const {
        cameras,
        thumbnailCamera,
        isLoading: isThumbnailCameraLoading,
        changeThumbnailCamera,
    } = useThumbnailCamera(dataset.id!);

    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });
    const environmentId = environments[0]?.id;
    const { data: environment } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        { params: { path: { project_id, environment_id: environmentId ?? '' } } },
        { enabled: environmentId !== undefined }
    );

    const { data: selectedEpisode, isLoading: isEpisodeLoading } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes/{episode_index}',
        {
            params: {
                path: {
                    dataset_id: String(dataset.id),
                    episode_index: Number(activeEpisodeIndex),
                },
            },
        },
        {
            enabled: activeEpisodeIndex !== null,
            placeholderData: keepPreviousData,
        }
    );

    const { defaultLayout, onLayoutChanged } = useDefaultLayout({
        id: 'dataset-viewer',
        onlySaveAfterUserInteractions: true,
    });

    if (episodes.length === 0) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <EmptyIllustration />
                    <Content>Currently there are no episodes.</Content>
                    <Text>It&apos;s time to begin recording a dataset. </Text>
                    <Heading>No episodes yet</Heading>
                    <View margin={'size-100'}>
                        <RecordDatasetButton project_id={dataset.project_id} dataset_id={dataset.id!}>
                            <Text>Start recording</Text>
                        </RecordDatasetButton>
                    </View>
                </IllustratedMessage>
            </Flex>
        );
    }
    return (
        <>
            {isPending && <Loading mode='overlay' />}
            <Group
                id='dataset-viewer'
                orientation='horizontal'
                defaultLayout={defaultLayout}
                onLayoutChanged={onLayoutChanged}
            >
                <Panel id='episode-viewer' minSize='30%' className={classes.viewerPanel}>
                    {isEpisodeLoading || !selectedEpisode ? (
                        <Loading />
                    ) : environment === undefined ? (
                        <IllustratedMessage>
                            <Heading>No environment available</Heading>
                            <Content>Create an environment to view robot overlays for this episode.</Content>
                        </IllustratedMessage>
                    ) : (
                        <EpisodeViewer episode={selectedEpisode} dataset={dataset} environment={environment} />
                    )}
                </Panel>
                <Separator className={classes.separator} />
                <Panel id='episode-list' defaultSize='35%' minSize='250px' maxSize='70%' className={classes.listPanel}>
                    <Flex justifyContent='space-between' alignItems='center' gap='size-100' marginBottom='size-200'>
                        {selectedEpisodes.length === 0 ? (
                            <RecordDatasetButton project_id={dataset.project_id} dataset_id={dataset.id!}>
                                <Add fill='white' style={{ marginRight: '4px' }} />
                                <Text>Add Episode</Text>
                            </RecordDatasetButton>
                        ) : (
                            <Flex gap='size-200' justifyContent='end'>
                                <ActionButton onPress={() => setSelectedEpisodes([])}>
                                    <Text>Clear selection</Text>
                                </ActionButton>
                                <DialogTrigger>
                                    <ActionButton>
                                        <Delete fill='white' />
                                    </ActionButton>
                                    <AlertDialog
                                        onPrimaryAction={async () => {
                                            const deletePromise = deleteEpisodes(selectedEpisodes);
                                            setSelectedEpisodes([]);

                                            await deletePromise;
                                        }}
                                        title='Delete episodes'
                                        variant='warning'
                                        primaryActionLabel='Delete'
                                        isPrimaryActionDisabled={isPending}
                                    >
                                        Are you sure you want to delete {selectedEpisodes.length} selected{' '}
                                        {pluralize(selectedEpisodes.length, 'episode', 'episodes')}?
                                    </AlertDialog>
                                </DialogTrigger>
                            </Flex>
                        )}
                        <Flex alignItems='center'>
                            <ThumbnailCameraMenu
                                cameras={cameras}
                                selectedKey={thumbnailCamera}
                                onChange={changeThumbnailCamera}
                            />
                            <ActionButton
                                isQuiet
                                aria-label={`Sort by episode number: ${sortOrder} first`}
                                onPress={() => setSortOrder((order) => (order === 'newest' ? 'oldest' : 'newest'))}
                            >
                                <SortUpDown />
                            </ActionButton>
                        </Flex>
                    </Flex>
                    <EpisodeList
                        episodes={episodes}
                        onSelect={setActiveEpisodeIndex}
                        currentEpisode={activeEpisodeIndex}
                        sortOrder={sortOrder}
                        thumbnailCamera={thumbnailCamera}
                        isThumbnailCameraLoading={isThumbnailCameraLoading}
                    />
                </Panel>
            </Group>
        </>
    );
};
