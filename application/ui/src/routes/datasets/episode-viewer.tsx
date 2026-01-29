import { useEffect, useRef } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, View, Well } from '@geti/ui';

import { SchemaEpisode, SchemaEpisodeVideo } from '../../api/openapi-spec';
import EpisodeChart from '../../components/episode-chart/episode-chart';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { TimelineControls } from './timeline-controls';
import { usePlayer } from './use-player';

import classes from './episode-viewer.module.scss';

const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];

interface VideoView {
    dataset_id: string;
    episodeIndex: number;
    cameraName: string;
    aspectRatio: number;
    time: number;
    episodeVideo: SchemaEpisodeVideo;
}
const VideoView = ({ cameraName, dataset_id, episodeIndex, aspectRatio, time, episodeVideo }: VideoView) => {
    const url = `/api/dataset/${dataset_id}/${episodeIndex}/${cameraName}.mp4`;

    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.currentTime = time + episodeVideo.start;
        }
    }, [time, episodeVideo.start]);

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

export const EpisodeViewer = ({ dataset_id, episode }: EpisodeViewerProps) => {
    const player = usePlayer(episode);
    const frameIndex = Math.floor(player.time * episode.fps);
    const cameras = Object.keys(episode.videos).map((m) => m.replace("observation.images.", ""))

    return (
        <RobotModelsProvider>
            <Flex direction={'column'} height={'100%'} position={'relative'}>
                <Flex direction={'row'} flex gap={'size-100'}>
                    <Flex direction={'column'} alignContent={'start'} flex gap={'size-30'}>
                        {cameras.map((camera) => (
                            <VideoView
                                key={camera}
                                aspectRatio={640 / 480}
                                cameraName={camera}
                                episodeIndex={episode.episode_index}
                                dataset_id={dataset_id}
                                time={player.time}
                                episodeVideo={episode.videos[`observation.images.${camera}`]}
                            />
                        ))}
                    </Flex>
                    <Flex flex={3} minWidth={0}>
                        <RobotViewer jointValues={episode.actions[frameIndex]} />
                    </Flex>
                </Flex>
                <div className={classes.timeline}>
                    <Disclosure isQuiet>
                        <DisclosureTitle>Timeline</DisclosureTitle>
                        <DisclosurePanel>
                            <EpisodeChart
                                actions={episode.actions}
                                joints={joints}
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
