import { useMemo, useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    Button,
    Content,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    IllustratedMessage,
    Loading,
    Text,
    View,
} from '@geti-ui/ui';
import { Add, Delete } from '@geti-ui/ui/icons';
import { keepPreviousData } from '@tanstack/react-query';

import { $api } from '../../api/client';
import { useDeleteEpisodeQuery } from '../../features/datasets/episodes/use-episodes';
import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';
import { pluralize } from '../../utils';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { useDataset } from './dataset-provider';
import { EpisodeList } from './episode-list';
import { EpisodeViewer } from './episode-viewer';

export const DatasetViewer = () => {
    const { dataset, episodes, selectedEpisodes, setSelectedEpisodes } = useDataset();
    const { project_id } = useProjectId();

    const { deleteEpisodes, isPending } = useDeleteEpisodeQuery(dataset.id!);
    const [currentEpisode, setCurrentEpisode] = useState<number | null>(null);

    if (episodes.length > 0 && currentEpisode === null) {
        setCurrentEpisode(episodes[0].episode_index);
    }

    const currentEpisodeIndex = useMemo(() => {
        if (currentEpisode !== null && episodes.some((episode) => episode.episode_index === currentEpisode)) {
            return currentEpisode;
        }

        return episodes[0]?.episode_index ?? null;
    }, [currentEpisode, episodes]);

    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: { path: { project_id, environment_id: dataset.environment_id } },
        }
    );

    const { data: selectedEpisode, isLoading: isEpisodeLoading } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes/{episode_index}',
        {
            params: {
                path: {
                    dataset_id: String(dataset.id),
                    episode_index: currentEpisodeIndex,
                },
            },
        },
        {
            placeholderData: keepPreviousData,
        }
    );

    const recordPath = paths.project.datasets.record({ project_id: dataset.project_id, dataset_id: dataset.id! });

    if (episodes.length === 0) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <EmptyIllustration />
                    <Content>Currently there are no episodes.</Content>
                    <Text>It&apos;s time to begin recording a dataset. </Text>
                    <Heading>No episodes yet</Heading>
                    <View margin={'size-100'}>
                        <Button href={recordPath} alignSelf='end' marginBottom={'size-200'}>
                            <Text>Start recording</Text>
                        </Button>
                    </View>
                </IllustratedMessage>
            </Flex>
        );
    }
    return (
        <Flex direction={'row'} height={'100%'} flex gap={'size-100'}>
            {isPending && <Loading mode='overlay' />}
            <View flex={1}>
                {isEpisodeLoading || !selectedEpisode ? (
                    <Loading />
                ) : (
                    <EpisodeViewer episode={selectedEpisode} dataset={dataset} environment={environment} />
                )}
            </View>
            <Divider orientation='vertical' size='S' />
            <Flex direction='column'>
                {selectedEpisodes.length === 0 ? (
                    <Button
                        href={recordPath}
                        variant='secondary'
                        alignSelf='end'
                        marginEnd='size-400'
                        marginBottom={'size-200'}
                    >
                        <Add fill='white' style={{ marginRight: '4px' }} />
                        <Text>Add Episode</Text>
                    </Button>
                ) : (
                    <Flex marginBottom='size-200' gap='size-200' justifyContent='end' marginEnd='size-400'>
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
                <EpisodeList
                    episodes={episodes}
                    onSelect={setCurrentEpisode}
                    currentEpisode={currentEpisodeIndex ?? -1}
                />
            </Flex>
        </Flex>
    );
};
