import { useEffect, useState } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Divider, Flex, Text } from '@geti-ui/ui';

import { SchemaDatasetOutput, SchemaEnvironmentWithRelations, SchemaEpisode } from '../../api/openapi-spec';
import EpisodeChart from '../../components/episode-chart/episode-chart';
import { EpisodeDockView } from '../../features/datasets/episodes/episode-dock-view';
import { EpisodeTag } from '../../features/datasets/episodes/episode-tag';
import {
    EpisodeViewerProvider,
    useEpisodeViewer,
} from '../../features/datasets/episodes/episode-viewer-provider.component';
import { Player } from '../../features/datasets/episodes/use-player';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { TimelineControls } from './timeline-controls';

import classes from './episode-viewer.module.css';

interface EpisodeViewerProps {
    episode: SchemaEpisode;
    dataset: SchemaDatasetOutput;
    environment: SchemaEnvironmentWithRelations;
}

interface LiveEpisodeChartProps {
    episode: SchemaEpisode;
    player: Player;
}
const LiveEpisodeChart = ({ episode, player }: LiveEpisodeChartProps) => {
    const [time, setTime] = useState<number>(player.timeRef.current);

    useEffect(() => {
        setTime(player.timeRef.current);
        if (player.isPlaying) {
            const interval = setInterval(() => {
                setTime(player.timeRef.current);
            }, 1000 / 60);
            return () => clearInterval(interval);
        }
    }, [player.timeRef, player.state, player.isPlaying]);

    const onSeek = (newTime: number) => {
        player.seek(newTime);
        setTime(newTime);
    };

    return (
        <EpisodeChart
            actions={episode.actions}
            joints={episode.action_keys}
            fps={episode.fps}
            time={time}
            seek={onSeek}
            isPlaying={player.isPlaying}
            play={player.play}
            pause={player.pause}
        />
    );
};

const EpisodeTimelineComponent = () => {
    const { player, episode } = useEpisodeViewer();

    return (
        <div className={classes.timeline}>
            <Disclosure isQuiet>
                <DisclosureTitle>Timeline</DisclosureTitle>
                <DisclosurePanel>
                    <LiveEpisodeChart episode={episode} player={player} />
                </DisclosurePanel>
            </Disclosure>
            {/* We reset the time controls state for the new episode */}
            <TimelineControls key={episode.episode_index} player={player} />
        </div>
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
                        <EpisodeDockView episode={episode} dataset={dataset} environment={environment} />
                    </Flex>
                    <EpisodeTimelineComponent />
                </Flex>
            </RobotModelsProvider>
        </EpisodeViewerProvider>
    );
};
