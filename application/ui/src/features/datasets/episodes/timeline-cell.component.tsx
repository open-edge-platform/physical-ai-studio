import EpisodeChart from "../../../components/episode-chart/episode-chart";
import { useEpisodeViewer } from "./episode-viewer-provider.component"

export const TimelineCell = () => {
    const { player, episode } = useEpisodeViewer();
    return <EpisodeChart
        actions={episode.actions}
        joints={episode.action_keys}
        fps={episode.fps}
        time={player.time}
        seek={player.seek}
        isPlaying={player.isPlaying}
        play={player.play}
        pause={player.pause}
    />
}
