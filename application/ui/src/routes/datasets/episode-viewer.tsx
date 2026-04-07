import { useEffect, useRef } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Divider, Flex, Text, View, Well } from '@geti/ui';

import { $api, fetchClient } from '../../api/client';
import { SchemaDatasetOutput, SchemaEpisode, SchemaEpisodeVideo } from '../../api/openapi-spec';
import EpisodeChart from '../../components/episode-chart/episode-chart';
import { EpisodeTag } from '../../features/datasets/episodes/episode-tag';
import { useProjectId } from '../../features/projects/use-project';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { TimelineControls } from './timeline-controls';

import classes from './episode-viewer.module.scss';
import { EpisodeDockView } from '../../features/datasets/episodes/episode-dock-view';
import { EpisodeViewerProvider, useEpisodeViewer } from '../../features/datasets/episodes/episode-viewer-provider.component';

interface EpisodeViewerProps {
    episode: SchemaEpisode;
    dataset: SchemaDatasetOutput;
}

const EpisodeTimelineComponent = () => {
    const { player, episode } = useEpisodeViewer();

    return (
        <div className={classes.timeline}>
            <Disclosure isQuiet>
                <DisclosureTitle>Timeline</DisclosureTitle>
                <DisclosurePanel>
                    <EpisodeChart
                        actions={episode.actions}
                        joints={episode.action_keys}
                        fps={episode.fps}
                        time={player.time}
                        seek={player.seek}
                        isPlaying={player.isPlaying}
                        play={player.play}
                        pause={player.pause}
                    />
                </DisclosurePanel>
            </Disclosure>
            <TimelineControls player={player} />
        </div>
    );
}

export const EpisodeViewer = ({ episode, dataset }: EpisodeViewerProps) => {
    const { project_id } = useProjectId();

    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: { path: { project_id, environment_id: dataset.environment_id } },
        }
    );

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
                        <EpisodeDockView episode={episode} dataset={dataset} environment={environment} />
                    </Flex>
                    <EpisodeTimelineComponent />
                </Flex>
            </RobotModelsProvider>
        </EpisodeViewerProvider >
    );
};
