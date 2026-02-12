import { useState } from 'react';

import { Button, Content, DialogTrigger, Divider, Flex, Heading, IllustratedMessage, Text, View } from '@geti/ui';
import { Add } from '@geti/ui/icons';

import { $api } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { TeleoperationSetupModal } from '../../features/configuration/teleoperation/teleoperation';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { EpisodeList } from './episode-list';
import { EpisodeViewer } from './episode-viewer';
import { useRecording } from './recording-provider';
import { RecordingViewer } from './recording-viewer';

interface DatasetViewerProps {
    dataset: SchemaDatasetOutput;
}

export const DatasetViewer = ({ dataset }: DatasetViewerProps) => {
    const { isRecording, recordingConfig, setRecordingConfig } = useRecording();
    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id: dataset.id!,
            },
        },
    });

    const [currentEpisode, setCurrentEpisode] = useState<number>(0);

    if (isRecording && recordingConfig) {
        return <RecordingViewer recordingConfig={recordingConfig} />;
    }

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
                <EpisodeViewer episode={episodes[currentEpisode]} dataset_id={dataset.id!} />
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
