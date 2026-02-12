import { useState } from 'react';

import { ActionButton, AlertDialog, Button, Content, DialogTrigger, Divider, Flex, Heading, IllustratedMessage, Loading, Text, View } from '@geti/ui';
import { Add, Delete } from '@geti/ui/icons';

import { TeleoperationSetupModal } from '../../features/configuration/teleoperation/teleoperation';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { EpisodeList } from './episode-list';
import { EpisodeViewer } from './episode-viewer';
import { useRecording } from './recording-provider';
import { RecordingViewer } from './recording-viewer';
import { useDataset } from './dataset-provider';

export const DatasetViewer = () => {
    const { isRecording, recordingConfig, setRecordingConfig } = useRecording();
    const { dataset, episodes, deleteEpisodes, selectedEpisodes, setSelectedEpisodes, isPending } = useDataset();

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
            { isPending && <Loading mode='overlay' />}
            <View flex={1}>
                <EpisodeViewer episode={episodes[currentEpisode]} dataset_id={dataset.id!} />
            </View>
            <Divider orientation='vertical' size='S' />
            <Flex direction='column'>
                {selectedEpisodes.length === 0
                    ? <DialogTrigger>
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
                    : <Flex marginBottom='size-200' gap="size-200" justifyContent='end' marginEnd='size-400'>
                          <Button variant='secondary' onPress={() => setSelectedEpisodes([])}>
                              <Text>Clear selection</Text>
                          </Button>
                          <DialogTrigger>
                            <ActionButton>
                                <Delete fill="white" />
                            </ActionButton>
                            <AlertDialog onPrimaryAction={() => deleteEpisodes(selectedEpisodes)}
                                title="Delete episodes"
                                variant="warning"
                                primaryActionLabel="Delete">
                                Are you sure you want to delete the selected episodes?
                            </AlertDialog>
                          </DialogTrigger>
                      </Flex>
                }
                <EpisodeList episodes={episodes} onSelect={setCurrentEpisode} currentEpisode={currentEpisode} />
            </Flex>
        </Flex>
    );
};
