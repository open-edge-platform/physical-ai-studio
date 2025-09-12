import { View, Text, Flex, Divider, Well, ListBox, Item, ListView, Selection } from '@geti/ui'
import { SchemaEpisode } from "../../api/openapi-spec"

interface EpisodeViewerProps {
    episodeIndex: number
    episode: SchemaEpisode
}
export const EpisodeViewer = ({episodeIndex, episode}: EpisodeViewerProps) => {
    return (
        <Flex>
            <Text>Viewer goes here for episode {episodeIndex}</Text>
        </Flex>
    )

}