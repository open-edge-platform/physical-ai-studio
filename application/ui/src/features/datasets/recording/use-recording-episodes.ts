import { $api } from '../../../api/client';
import { useRuntimeSession } from '../../robots/runtime-session-provider';

interface RecordingEpisodes {
    episodesInDataset: number;
    episodesThisSession: number;
    nextEpisodeNumber: number;
}

export const useRecordingEpisodes = (): RecordingEpisodes => {
    const { dataset, state } = useRuntimeSession();

    if (dataset?.id === undefined) {
        throw new Error('Cannot load recording episodes without a dataset.');
    }

    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: { path: { dataset_id: dataset.id } },
    });

    const episodesThisSession = state.episodes_recorded;
    const episodesInDataset = episodes.length + episodesThisSession;

    return { episodesInDataset, episodesThisSession, nextEpisodeNumber: episodesInDataset + 1 };
};
