import { useState } from 'react';

import { Flex, Item, ListView, Selection, View } from '@geti/ui';

import { EpisodeViewer } from './episode-viewer';
import { $api } from '../../api/client';

interface DatasetViewerProps {
    id: string;
}
export const DatasetViewer = ({ id: dataset_id }: DatasetViewerProps) => {
    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id
            }
        }
    })

    const [episodeIndexKey, setEpisodeIndexKey] = useState<Selection>(new Set([0]));
    const [currentEpisode] = episodeIndexKey as Set<number>;
    const items = episodes.map((_, index) => ({ id: index, name: `Episode ${index + 1}` }));


    return (
        <Flex direction={'row'}>
            <View maxWidth={'size-2000'} flex={1}>
                <ListView
                    selectedKeys={episodeIndexKey}
                    selectionMode={'single'}
                    items={items}
                    selectionStyle='highlight'
                    onSelectionChange={setEpisodeIndexKey}
                >
                    {(item) => <Item>{item.name}</Item>}
                </ListView>
            </View>
            {currentEpisode !== undefined && (
                <View flex={1}>
                    <EpisodeViewer episode={episodes[currentEpisode]} />
                </View>
            )}
        </Flex>
    );
};
