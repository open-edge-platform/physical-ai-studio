import { useState } from 'react';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';

import { Divider, Flex, Button, Content, Heading, Text, View, DialogTrigger, IllustratedMessage } from '@geti/ui';

import { $api } from '../../api/client';
import { SchemaDatasetOutput, SchemaEpisode } from '../../api/openapi-spec';
import { EpisodeViewer } from './episode-viewer';
import { useRecording } from './recording-provider';
import { RecordingViewer } from './recording-viewer';
import { EpisodeList } from './episode-list';
import { Add } from '@geti/ui/icons';
import { TeleoperationSetupModal } from '../../features/configuration/teleoperation/teleoperation';

interface DatasetViewerProps {
    dataset: SchemaDatasetOutput;
}

export const DatasetViewer = ({ dataset }: DatasetViewerProps) => {
    const { isRecording, recordingConfig, setRecordingConfig } = useRecording();
    const { data: existingEpisodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id: dataset.id!,
            },
        },
    });

    const [episodes, setEpisodes] = useState<SchemaEpisode[]>(existingEpisodes);

    const [currentEpisode, setCurrentEpisode] = useState<number>(0);

    const showEpisodeViewer = !isRecording;

    const addEpisode = (episode: SchemaEpisode) => {
        setEpisodes((current) => [...current, episode]);
        setCurrentEpisode(episode.episode_index);
    };

    if (episodes.length === 0 && !isRecording) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <EmptyIllustration />
                    <Content> Currently there are episodes. </Content>
                    <Text>It&apos;s time to begin recording a dataset. </Text>
                    <Heading>No episodes yet</Heading>
                    <View margin={'size-100'}>
                        <DialogTrigger>
                            <Button variant='secondary' alignSelf='end' marginEnd='size-400' marginBottom={'size-200'}>
                                <Text>Start recording</Text>
                            </Button>
                            {(close) =>
                                TeleoperationSetupModal((config) => {
                                    setRecordingConfig(config);
                                    close();
                                }, dataset!)
                            }
                        </DialogTrigger>
                    </View>
                </IllustratedMessage>
            </Flex>
        );
    }
    return (
        <Flex direction={'row'} height={'100%'} flex gap={'size-100'}>
            <View flex={1}>
                {showEpisodeViewer ? (
                    <EpisodeViewer episode={episodes[currentEpisode]} dataset_id={dataset.id!} />
                ) : (
                    <RecordingViewer recordingConfig={recordingConfig!} addEpisode={addEpisode} />
                )}
            </View>
            <Divider orientation='vertical' size='S' />
            <Flex direction='column'>
                <DialogTrigger>
                    <Button variant='secondary' alignSelf='end' marginEnd='size-400' marginBottom={'size-200'}>
                        <Add fill='white' style={{ marginRight: '4px' }} />
                        <Text>Add Episode</Text>
                    </Button>
                    {(close) =>
                        TeleoperationSetupModal((config) => {
                            setRecordingConfig(config);
                            close();
                        }, dataset!)
                    }
                </DialogTrigger>
                <EpisodeList episodes={episodes} onSelect={setCurrentEpisode} currentEpisode={currentEpisode} />
            </Flex>
        </Flex>
    );
};
