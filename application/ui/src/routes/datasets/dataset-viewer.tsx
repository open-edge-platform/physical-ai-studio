import { useState } from 'react';

import { Divider, Flex, Item, ListView, Selection, Text, View } from '@geti/ui';

import { $api } from '../../api/client';
import { SchemaEpisode } from '../../api/openapi-spec';
import { EpisodeViewer } from './episode-viewer';
import { useRecording } from './recording-provider';
import { RecordingViewer } from './recording-viewer';
import { EpisodeList } from './episode-list';

interface DatasetViewerProps {
    id: string;
}
export const DatasetViewer = ({ id: dataset_id }: DatasetViewerProps) => {
    const { isRecording, recordingConfig } = useRecording();
    const { data: existingEpisodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id,
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

    if (currentEpisode === undefined && !isRecording) {
        return (
            <View>
                <Text>No episodes yet... record one</Text>
            </View>
        );
    }
    return (
        <Flex direction={'row'} height={'100%'} flex gap={'size-100'}>
            <View flex={1}>
                {showEpisodeViewer ? (
                    <EpisodeViewer episode={episodes[currentEpisode]} dataset_id={dataset_id} />
                ) : (
                    <RecordingViewer recordingConfig={recordingConfig!} addEpisode={addEpisode} />
                )}
            </View>
            <Divider orientation='vertical' size='S' />
            <EpisodeList episodes={episodes} onSelect={setCurrentEpisode} currentEpisode={currentEpisode} />
        </Flex>
    );
};
