import { useState } from 'react';

import { Flex, Item, ListView, Selection, View } from '@geti/ui';

import { useProject } from '../projects/project.provider';
import { EpisodeViewer } from './episode-viewer';
import { useDataset } from './use-dataset';

interface DatasetViewerProps {
    repo_id: string;
}
export const DatasetViewer = ({ repo_id }: DatasetViewerProps) => {
    const { project } = useProject();
    const { dataset } = useDataset(project.id, repo_id);
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
