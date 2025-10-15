import { Flex, Text } from '@geti/ui';

import { SchemaEpisode } from '../../api/openapi-spec';
import EpisodeChart from '../../components/episode-chart/episode-chart';
import { useProject } from '../../features/projects/use-project';
import { useParams } from 'react-router';

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
        <Flex flex UNSAFE_style={{ position: 'relative' }}>
            <video autoPlay src={url} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
        </Flex>
    )
};


export const EpisodeViewer = ({ dataset_id, episode }: EpisodeViewerProps) => {
    const project = useProject();
    return (
        <Flex direction={'column'} height={'100%'}>
            <Flex direction={'row'} flex>
                <Flex direction={'column'} justifyContent={'space-evenly'} flex gap={'size-30'}>
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
                <Text flex>Simulator go here</Text>
            </Flex>
            <Flex>
                <EpisodeChart actions={episode.actions} joints={joints} fps={episode.fps} />
            </Flex>
        </Flex>
    );
};
