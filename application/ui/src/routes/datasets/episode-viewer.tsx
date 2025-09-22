import { Flex } from '@geti/ui';

import { SchemaEpisode } from '../../api/openapi-spec';
import EpisodeChart from '../../components/episode-chart/episode-chart';

interface EpisodeViewerProps {
    episode: SchemaEpisode;
}

const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];

export const EpisodeViewer = ({ episode }: EpisodeViewerProps) => {
    return (
        <Flex>
            <EpisodeChart actions={episode.actions} joints={joints} fps={episode.fps} />
        </Flex>
    );
};
