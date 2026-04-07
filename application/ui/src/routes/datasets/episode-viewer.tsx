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

interface VideoView {
    dataset_id: string;
    cameraName: string;
    aspectRatio: number;
    time: number;
    episodeVideo: SchemaEpisodeVideo;
}
const VideoView = ({ dataset_id, cameraName, aspectRatio, time, episodeVideo }: VideoView) => {
    const url = fetchClient.PATH('/api/dataset/{dataset_id}/video/{video_path}', {
        params: {
            path: {
                dataset_id,
                video_path: episodeVideo.path,
            },
        },
    });

    const videoRef = useRef<HTMLVideoElement>(null);

    // Make sure webpage renders when video doesn't load correctly
    useEffect(() => {
        const video = videoRef.current;
        const start = episodeVideo?.start;

        if (!video) return;
        if (video.readyState < 1) return;
        if (!Number.isFinite(time) || !Number.isFinite(start)) return;

        video.currentTime = time + start;
    }, [time, episodeVideo?.start]);

    /* eslint-disable jsx-a11y/media-has-caption */
    return (
        <Flex UNSAFE_style={{ aspectRatio }}>
            <Well flex UNSAFE_style={{ position: 'relative' }}>
                <View height={'100%'} position={'relative'}>
                    <video ref={videoRef} src={url} className={classes.cameraVideo} />
                </View>
                <div className={classes.cameraTag}> {cameraName} </div>
            </Well>
        </Flex>
    );
};

interface EpisodeViewerProps {
    episode: SchemaEpisode;
    dataset: SchemaDatasetOutput;
}

const EpisodeTimelineComponent = () => {
    const { player } = useEpisodeViewer();
    return <TimelineControls player={player} />
}

export const EpisodeViewer = ({ episode, dataset }: EpisodeViewerProps) => {
    return (
        <EpisodeViewerProvider episode={episode}>
            <RobotModelsProvider>
                <Flex direction={'column'} height={'100%'} position={'relative'}>
                    <Flex gap='size-100' marginBottom='size-100'>
                        <EpisodeTag episode={episode} variant='medium' />
                        <Divider orientation='vertical' size='S' />
                        <Text>{episode.tasks.join(', ')}</Text>
                    </Flex>
                    <Flex direction={'row'} flex gap={'size-100'}>
                        <EpisodeDockView episode={episode} dataset={dataset} />
                    </Flex>
                    <div className={classes.timeline}>
                        <EpisodeTimelineComponent />
                    </div>
                </Flex>
            </RobotModelsProvider>
        </EpisodeViewerProvider>
    );
};
