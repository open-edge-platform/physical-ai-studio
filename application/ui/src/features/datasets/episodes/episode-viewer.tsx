import { Divider, Flex, Text } from '@geti-ui/ui';

import { SchemaDatasetOutput, SchemaEnvironmentWithRelations, SchemaEpisode } from '../../../api/openapi-spec';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { EpisodeDockView } from './episode-dock-view';
import { EpisodeTag } from './episode-tag';
import { EpisodeViewerProvider, useEpisodeViewer } from './episode-viewer-provider.component';
import { TrimTimeline } from './trim-episode/trim-timeline';

interface EpisodeViewerProps {
    episode: SchemaEpisode;
    dataset: SchemaDatasetOutput;
    environment: SchemaEnvironmentWithRelations;
}

const EpisodeTimeline = ({ datasetId }: { datasetId: string }) => {
    const { player, episode } = useEpisodeViewer();

    return (
        <TrimTimeline
            // Remount per episode (and after a trim) to reset the range and playhead
            key={`${episode.episode_index}-${episode.length}`}
            datasetId={datasetId}
            episode={episode}
            player={player}
        />
    );
};

export const EpisodeViewer = ({ episode, dataset, environment }: EpisodeViewerProps) => {
    return (
        <EpisodeViewerProvider episode={episode} environment={environment}>
            <RobotModelsProvider>
                <Flex direction={'column'} height={'100%'} position={'relative'}>
                    <Flex gap='size-100' marginBottom='size-100'>
                        <EpisodeTag episode={episode} variant='medium' />
                        <Divider orientation='vertical' size='S' />
                        <Text>{episode.tasks.join(', ')}</Text>
                    </Flex>
                    <Flex direction={'row'} flex gap={'size-100'}>
                        <EpisodeDockView
                            key={dataset.id}
                            episode={episode}
                            dataset={dataset}
                            environment={environment}
                        />
                    </Flex>
                    <EpisodeTimeline datasetId={String(dataset.id)} />
                </Flex>
            </RobotModelsProvider>
        </EpisodeViewerProvider>
    );
};
