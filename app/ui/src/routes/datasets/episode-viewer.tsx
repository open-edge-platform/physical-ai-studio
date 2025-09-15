import { View, Text, Flex, Divider, Well, ListBox, Item, ListView, Selection } from '@geti/ui'
import { SchemaEpisode } from "../../api/openapi-spec"
import EpisodeChart from '../../components/episode-chart/episode-chart'

interface EpisodeViewerProps {
    episodeIndex: number
    episode: SchemaEpisode
}

const joints = [
    "shoulder_pan", 
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
];

export const EpisodeViewer = ({episodeIndex, episode}: EpisodeViewerProps) => {
    return (
        <Flex>
            <EpisodeChart actions={episode.actions} joints={joints} fps={episode.fps}/>
        </Flex>
    )

}