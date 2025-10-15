import { useState } from 'react';

import { Flex, Item, ListView, Selection, View, Text, Divider } from '@geti/ui';

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


    if (currentEpisode === undefined) {
        return (
            <View>
                <Text>No episodes yet... record one</Text>
            </View>
        )
    }
    return (
        <Flex direction={'row'} height="100%" flex gap={'size-100'}>
            <View flex={1}>
                <EpisodeViewer episode={episodes[currentEpisode]} dataset_id={dataset_id} />
            </View>
            <Divider orientation='vertical' size='S'/>
            <Flex
                flex
                direction="column"
                maxWidth="size-2000"
            >
                <ListView
                    disallowEmptySelection
                    selectedKeys={episodeIndexKey}
                    selectionMode="single"
                    items={items}
                    selectionStyle="highlight"
                    onSelectionChange={setEpisodeIndexKey}
                    flex={'1 0 0'}
                >
                    {(item) => <Item>{item.name}</Item>}
                </ListView>
            </Flex>
        </Flex>
    );
};
