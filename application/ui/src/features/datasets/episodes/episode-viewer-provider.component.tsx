import { ReactNode, useContext } from "react"
import { createContext } from "react";
import { SchemaEnvironmentWithRelations, SchemaEpisode } from "../../../api/openapi-spec";
import { Player, usePlayer } from "./use-player";


type EpisodeViewerContextValue = null | {
    player: Player;
    episode: SchemaEpisode;
    environment: SchemaEnvironmentWithRelations;
};

const EpisodeViewerContext = createContext<EpisodeViewerContextValue>(null);

interface EpisodeViewerProviderProps {
    children: ReactNode;
    episode: SchemaEpisode;
    environment: SchemaEnvironmentWithRelations;
}

export const EpisodeViewerProvider = ({ children, episode, environment }: EpisodeViewerProviderProps) => {
    const player = usePlayer(episode);

    return <EpisodeViewerContext.Provider
        value={{
            player,
            episode,
            environment,
        }}>
        {children}
    </EpisodeViewerContext.Provider>

}


export const useEpisodeViewer = () => {
    const ctx = useContext(EpisodeViewerContext);
    if (!ctx) throw new Error('useEpisodeViewer must be used within EpisodeViewerProvider');
    return ctx;
};
