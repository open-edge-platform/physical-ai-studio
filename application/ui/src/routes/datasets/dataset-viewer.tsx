import { useEffect, useMemo, useState } from 'react';

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
import { useQuery } from '@tanstack/react-query';

import { SchemaEpisode } from '../../api/openapi-spec';
import { useDeleteEpisodeQuery } from '../../features/datasets/episodes/use-episodes';
import { paths } from '../../router';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { useDataset } from './dataset-provider';
import { EpisodeList } from './episode-list';
import { EpisodeViewer } from './episode-viewer';

export const DatasetViewer = () => {
    const { dataset, episodes, selectedEpisodes, setSelectedEpisodes } = useDataset();

    const { deleteEpisodes, isPending } = useDeleteEpisodeQuery(dataset.id!);
    const [currentEpisode, setCurrentEpisode] = useState<number | null>(null);

    useEffect(() => {
        if (episodes.length > 0 && currentEpisode === null) {
            setCurrentEpisode(episodes[0].episode_index);
        }
    }, [episodes, currentEpisode]);

    const currentEpisodeIndex = useMemo(() => {
        if (currentEpisode !== null && episodes.some((episode) => episode.episode_index === currentEpisode)) {
            return currentEpisode;
        }

        return episodes[0]?.episode_index ?? null;
    }, [currentEpisode, episodes]);

    const { data: selectedEpisode, isLoading: isEpisodeLoading } = useQuery({
        queryKey: ['dataset-episode', dataset.id, currentEpisodeIndex],
        enabled: currentEpisodeIndex !== null,
        queryFn: async (): Promise<SchemaEpisode> => {
            const response = await fetch(
                `/api/dataset/${encodeURIComponent(dataset.id!)}/episodes/${encodeURIComponent(currentEpisodeIndex!)}`
            );

            if (!response.ok) {
                throw new Error(`Failed to load episode ${currentEpisodeIndex}`);
            }

            return (await response.json()) as SchemaEpisode;
        },
    });

    const recordPath = paths.project.datasets.record({ project_id: dataset.project_id, dataset_id: dataset.id! });

    if (episodes.length === 0) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <EmptyIllustration />
                    <Content> Currently there are episodes. </Content>
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
                    <EpisodeViewer episode={selectedEpisode} dataset={dataset} />
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
                                onPrimaryAction={() => deleteEpisodes(selectedEpisodes)}
                                title='Delete episodes'
                                variant='warning'
                                primaryActionLabel='Delete'
                            >
                                Are you sure you want to delete the selected episodes?
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
