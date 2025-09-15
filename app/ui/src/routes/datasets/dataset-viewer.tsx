import { useState } from "react";
import { useDataset } from "./dataset.provider"
import { View, Text, Flex, Divider, Well, ListBox, Item, ListView, Selection } from '@geti/ui'
import { EpisodeViewer } from "./episode-viewer";


interface Key {
    currentKey: number
}
export const DatasetViewer = () => {
    const { dataset } = useDataset();
    const [episodeIndexKey, setEpisodeIndexKey] = useState<Selection>(new Set([0]));
    const [currentEpisode] = (episodeIndexKey as Set<number>);
    const episodes = dataset.episodes.map((m, index) => ({ id: index, name: `Episode ${index + 1}` }));

    return (
        <Flex direction={'row'}>
            <View maxWidth={"size-2000"} flex={1}>
                <ListView
                    selectedKeys={episodeIndexKey}
                    selectionMode={'single'}
                    items={episodes}
                    selectionStyle="highlight"
                    onSelectionChange={setEpisodeIndexKey}>
                    {(item) => <Item>{item.name}</Item>}
                </ListView>
            </View>
            {currentEpisode !== undefined && (
                <View flex={1}>
                    <EpisodeViewer episodeIndex={currentEpisode} episode={dataset.episodes[currentEpisode]} />
                </View>
            )}
        </Flex>
    )
}