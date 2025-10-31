import { useState } from 'react';

import { Divider, Flex, Item, ListView, Selection, Text, View } from '@geti/ui';

import { $api } from '../../api/client';
import { EpisodeViewer } from './episode-viewer';
import { SchemaEpisode, SchemaTeleoperationConfig } from '../../api/openapi-spec';
import { RecordingViewer } from './recording-viewer';
import { EpisodesProvider } from './episode-provider';

interface DatasetViewerProps {
    id: string;
    recordingConfig: SchemaTeleoperationConfig | undefined;
}
export const DatasetViewer = ({ id: dataset_id, recordingConfig }: DatasetViewerProps) => {
    const { data: existingEpisodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

    const [episodes, setEpisodes] = useState<SchemaEpisode[]>(existingEpisodes);

    const [episodeIndexKey, setEpisodeIndexKey] = useState<Selection>(new Set(episodes.length > 0 ? [0] : undefined));
    const [currentEpisode] = episodeIndexKey as Set<number>;

    const items = episodes.map((_, index) => ({ id: index, name: `Episode ${index + 1}` })).toReversed();

    const addEpisode = (episode: SchemaEpisode) => {
        setEpisodes((current) => [...current, episode])
    }

    if (currentEpisode === undefined && recordingConfig === undefined) {
        return (
            <View>
                <Text>No episodes yet... record one</Text>
            </View>
        );
    }
    return (
        <Flex direction={'row'} height='100%' flex gap={'size-100'}>
            <View flex={1}>
                {recordingConfig === undefined
                    ? <EpisodeViewer episode={episodes[currentEpisode]} dataset_id={dataset_id} />
                    : <RecordingViewer recordingConfig={recordingConfig} addEpisode={addEpisode} />
                }
            </View>
            <Divider orientation='vertical' size='S' />
            <Flex flex direction='column' maxWidth='size-2000'>
                <ListView
                    disallowEmptySelection
                    selectedKeys={episodeIndexKey}
                    selectionMode='single'
                    items={items}
                    selectionStyle='highlight'
                    onSelectionChange={setEpisodeIndexKey}
                    flex={'1 0 0'}
                >
                    {(item) => <Item>{item.name}</Item>}
                </ListView>
            </Flex>
        </Flex>
    );
};
