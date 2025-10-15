import { Flex, View, Text, Well, Disclosure, DisclosurePanel, DisclosureTitle, ActionButton } from '@geti/ui';

import { SchemaEpisode } from '../../api/openapi-spec';
import EpisodeChart from '../../components/episode-chart/episode-chart';
import { useProject } from '../../features/projects/use-project';

import classes from './episode-viewer.module.scss'
import { useEffect, useRef, useState } from 'react';
import { usePlayer } from './use-player';
import { TimelineControls } from './timeline-controls';
import RobotRenderer from '../../components/robot-renderer/robot-renderer';

const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];

interface VideoView {
    dataset_id: string;
    episodeIndex: number;
    cameraName: string;
    aspectRatio: number;
    time: number;
}
const VideoView = ({ cameraName, dataset_id, episodeIndex, aspectRatio, time }: VideoView) => {
    const url = `/api/dataset/${dataset_id}/${episodeIndex}/${cameraName}.mp4`;

    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.currentTime = time;
        }
    }, [time])

    return (
        <Flex UNSAFE_style={{ aspectRatio }}>
            <Well flex UNSAFE_style={{ position: 'relative' }}>
                <View height={'100%'} position={'relative'}>
                    <video ref={videoRef} src={url} className={classes.cameraVideo} />
                </View>
                <div className={classes.cameraTag}> {cameraName} </div>
            </Well>
        </Flex>
    )
};

interface EpisodeViewerProps {
    episode: SchemaEpisode;
    dataset_id: string
}

export const EpisodeViewer = ({ dataset_id, episode }: EpisodeViewerProps) => {
    const project = useProject();
    const player = usePlayer(episode);

    return (
        <Flex direction={'column'} height={'100%'} position={'relative'}>
            <Flex direction={'row'} flex gap={'size-100'}>
                <Flex direction={'column'} alignContent={'start'} flex gap={'size-30'}>
                    {project.config!.cameras.map((camera) => (
                        <VideoView
                            key={camera.name}
                            aspectRatio={camera.width / camera.height}
                            cameraName={camera.name}
                            episodeIndex={episode.episode_index}
                            dataset_id={dataset_id}
                            time={player.time}
                        />
                    ))}
                </Flex>
                <Flex flex={3} minWidth={0}>
                    <RobotRenderer episode={episode} robot_urdf_path='/SO101/so101_new_calib.urdf' time={player.time}/>
                </Flex>
            </Flex>
            <div className={classes.timeline}>
                <Disclosure isQuiet>
                    <DisclosureTitle>Timeline</DisclosureTitle>
                    <DisclosurePanel>
                        <EpisodeChart actions={episode.actions} joints={joints} fps={episode.fps} />
                    </DisclosurePanel>
                </Disclosure>

                <TimelineControls player={player} />
            </div>
        </Flex>
    );
};
