import { useState } from 'react';

import { Flex, Item, ListView, Selection, View, Text } from '@geti/ui';

import { $api } from '../../api/client';
import { EpisodeViewer } from './episode-viewer';

interface DatasetViewerProps {
    id: string;
}
export const DatasetViewer = ({ id: dataset_id }: DatasetViewerProps) => {
    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

    const [episodeIndexKey, setEpisodeIndexKey] = useState<Selection>(new Set(episodes.length > 0 ? [0] : undefined));
    const [currentEpisode] = episodeIndexKey as Set<number>;
    const items = episodes.map((_, index) => ({ id: index, name: `Episode ${index + 1}` }));

    return (
        <Flex direction={'row'}>
            {currentEpisode !== undefined 
            ? (
                <>
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

                <View flex={1}>
                    <EpisodeViewer episode={episodes[currentEpisode]} />
                </View>
                </>
            ) 
            : (

                <View>
                    <Text>No episodes yet... record one</Text>
                </View>
            )
            }
        </Flex>
    );
};
