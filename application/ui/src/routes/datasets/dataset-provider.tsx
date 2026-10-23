import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useEffect, useState } from 'react';

import { $api } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { getPendingRecordedEpisodes, PendingRecordedEpisodes } from '../../features/datasets/pending-recorded-episodes';

type EpisodeSummary = {
    episode_index: number;
    tasks: string[];
    length: number;
    fps: number;
};

type DatasetContextValue = null | {
    dataset_id: string;
    dataset: SchemaDatasetOutput;
    episodes: EpisodeSummary[];
    setSelectedEpisodes: Dispatch<SetStateAction<number[]>>;
    selectedEpisodes: number[];
};
const DatasetContext = createContext<DatasetContextValue>(null);

const EPISODES_REFETCH_INTERVAL_MS = 1000;
const EPISODES_POLLING_TIMEOUT_MS = 60_000;

type PendingEpisodesPolling = PendingRecordedEpisodes & {
    expiresAt: number;
};

interface DatasetProviderProps {
    dataset_id: string;
    children: ReactNode;
}
export const DatasetProvider = ({ dataset_id, children }: DatasetProviderProps) => {
    const [selectedEpisodes, setSelectedEpisodes] = useState<number[]>([]);
    const [pendingRecordedEpisodes, setPendingRecordedEpisodes] = useState<PendingEpisodesPolling | undefined>();

    useEffect(() => {
        const pending = getPendingRecordedEpisodes(dataset_id);
        setPendingRecordedEpisodes(
            pending === undefined ? undefined : { ...pending, expiresAt: Date.now() + EPISODES_POLLING_TIMEOUT_MS }
        );
        setSelectedEpisodes([]);
    }, [dataset_id]);

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

    const { data: episodes } = $api.useSuspenseQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes',
        {
            params: {
                path: {
                    dataset_id,
                },
            },
        },
        {
            // Recording finalizes asynchronously when its runtime session closes.
            // Revalidate cached episodes when returning to this page. If an
            // expected count was recorded, poll until finalization reaches it.
            refetchOnMount: 'always',
            refetchInterval: (query) =>
                pendingRecordedEpisodes !== undefined &&
                Date.now() < pendingRecordedEpisodes.expiresAt &&
                (query.state.data?.length ?? 0) < pendingRecordedEpisodes.expectedEpisodeCount
                    ? EPISODES_REFETCH_INTERVAL_MS
                    : false,
        }
    );

    return (
        <DatasetContext.Provider
            value={{
                dataset_id,
                dataset,
                episodes,
                setSelectedEpisodes,
                selectedEpisodes,
            }}
        >
            {children}
        </DatasetContext.Provider>
    );
};

export const useDataset = () => {
    const ctx = useContext(DatasetContext);
    if (!ctx) throw new Error('useDataset must be used within DatasetProvider');
    return ctx;
};

export type { EpisodeSummary };
