import { useEffect, useRef } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Divider, Flex, Text, View, Well } from '@geti/ui';

import { SchemaEpisode, SchemaEpisodeVideo } from '../../api/openapi-spec';
import EpisodeChart from '../../components/episode-chart/episode-chart';
import { EpisodeTag } from '../../features/datasets/episodes/episode-tag';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { TimelineControls } from './timeline-controls';
import { usePlayer } from './use-player';

import classes from './episode-viewer.module.scss';

interface VideoView {
    dataset_id: string;
    cameraName: string;
    aspectRatio: number;
    time: number;
    episodeVideo: SchemaEpisodeVideo;
}
const VideoView = ({ dataset_id, cameraName, aspectRatio, time, episodeVideo }: VideoView) => {
    const url = `/api/dataset/${dataset_id}/video/${episodeVideo.path}`;

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
    dataset_id: string;
}

export const EpisodeViewer = ({ episode, dataset_id }: EpisodeViewerProps) => {
    const player = usePlayer(episode);
    const frameIndex = Math.floor(player.time * episode.fps);
    const cameras = Object.keys(episode.videos).map((m) => m.replace('observation.images.', ''));
    const follower_robot_type = episode.follower_robot_types?.[0] ?? 'SO101_Follower';

    return (
        <RobotModelsProvider>
            <Flex direction={'column'} height={'100%'} position={'relative'}>
                <Flex gap='size-100' marginBottom='size-100'>
                    <EpisodeTag episode={episode} variant='medium' />
                    <Divider orientation='vertical' size='S' />
                    <Text>{episode.tasks.join(', ')}</Text>
                </Flex>
                <Flex direction={'row'} flex gap={'size-100'}>
                    <Flex direction={'column'} alignContent={'start'} flex gap={'size-30'}>
                        {cameras.map((camera) => (
                            <VideoView
                                key={camera}
                                dataset_id={dataset_id}
                                aspectRatio={640 / 480}
                                cameraName={camera}
                                time={player.time}
                                episodeVideo={episode.videos[`observation.images.${camera}`]}
                            />
                        ))}
                    </Flex>
                    <Flex flex={3} minWidth={0}>
                        <RobotViewer
                            featureValues={episode.actions[frameIndex]}
                            featureNames={episode.action_keys}
                            robotType={follower_robot_type}
                        />
                    </Flex>
                </Flex>
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
            </Flex>
        </RobotModelsProvider>
    );
};
