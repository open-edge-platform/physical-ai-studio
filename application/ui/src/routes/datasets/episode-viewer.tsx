import { Flex, View, Text, Well, Disclosure, DisclosurePanel, DisclosureTitle } from '@geti/ui';

import { SchemaEpisode } from '../../api/openapi-spec';
import EpisodeChart from '../../components/episode-chart/episode-chart';
import { useProject } from '../../features/projects/use-project';
import { useParams } from 'react-router';

import classes from './episode-viewer.module.scss'
import { useState } from 'react';

interface EpisodeViewerProps {
    episode: SchemaEpisode;
    dataset_id: string
}

const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];

interface VideoView {
    dataset_id: string;
    episodeIndex: number;
    cameraName: string;
    aspectRatio: number;
}
const VideoView = ({ cameraName, dataset_id, episodeIndex, aspectRatio }: VideoView) => {
    const url = `/api/dataset/${dataset_id}/${episodeIndex}/${cameraName}.mp4`;

    return (
        <Flex UNSAFE_style={{aspectRatio}}>
            <Well flex UNSAFE_style={{position: 'relative'}}>
                <View height={'100%'} position={'relative'}>
                    <video autoPlay src={url} className={classes.cameraVideo} />
                </View>
                <div className={classes.cameraTag}> {cameraName} </div>
            </Well>
        </Flex>
    )
};


export const EpisodeViewer = ({ dataset_id, episode }: EpisodeViewerProps) => {
    const project = useProject();
    const [showTimeline, setShowTimeline] = useState<boolean>(false);
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
                        />
                    ))}
                </Flex>
                <Flex flex={3} alignItems={'center'} justifyContent={'center'}>
                    <Text>Simulator go here</Text>
                </Flex>
            </Flex>
            <div className={classes.timeline}>
                <Disclosure isQuiet>
                    <DisclosureTitle>Timeline</DisclosureTitle>
                    <DisclosurePanel>
                        <EpisodeChart actions={episode.actions} joints={joints} fps={episode.fps} />
                    </DisclosurePanel>
                </Disclosure>
                Play pause and stuff
            </div>
        </Flex>
    );
};
