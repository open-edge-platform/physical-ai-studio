import { useState } from 'react';

import { Flex, Item, ListView, Selection, View } from '@geti/ui';

import { useDataset } from './dataset.provider';
import { EpisodeViewer } from './episode-viewer';

export const DatasetViewer = () => {
    const { dataset } = useDataset();
    const [episodeIndexKey, setEpisodeIndexKey] = useState<Selection>(new Set([0]));
    const [currentEpisode] = episodeIndexKey as Set<number>;
    const episodes = dataset.episodes.map((_, index) => ({ id: index, name: `Episode ${index + 1}` }));

    return (
        <Flex direction={'row'}>
            <View maxWidth={'size-2000'} flex={1}>
                <ListView
                    selectedKeys={episodeIndexKey}
                    selectionMode={'single'}
                    items={episodes}
                    selectionStyle='highlight'
                    onSelectionChange={setEpisodeIndexKey}
                >
                    {(item) => <Item>{item.name}</Item>}
                </ListView>
            </View>
            {currentEpisode !== undefined && (
                <View flex={1}>
                    <EpisodeViewer episode={dataset.episodes[currentEpisode]} />
                </View>
            )}
        </Flex>
    );
};
