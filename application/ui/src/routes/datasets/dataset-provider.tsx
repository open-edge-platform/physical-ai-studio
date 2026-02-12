import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../api/client';
import { SchemaDatasetOutput, SchemaEpisode } from '../../api/openapi-spec';

type DatasetContextValue = null | {
    dataset_id: string;
    dataset: SchemaDatasetOutput;
    episodes: SchemaEpisode[];
    deleteEpisodes: (episodeIndices: number[]) => void;
    setSelectedEpisodes: Dispatch<SetStateAction<number[]>>;
    selectedEpisodes: number[];
    isPending: boolean;
};
const DatasetContext = createContext<DatasetContextValue>(null);

interface DatasetProviderProps {
    dataset_id: string;
    children: ReactNode;
}
export const DatasetProvider = ({ dataset_id, children }: DatasetProviderProps) => {
    const queryClient = useQueryClient();
    const [selectedEpisodes, setSelectedEpisodes] = useState<number[]>([]);

    const { data: dataset, isPending: datasetPending } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

    const { data: episodes, isPending: episodesPending } = $api.useSuspenseQuery(
        'get',
        '/api/dataset/{dataset_id}/episodes',
        {
            params: {
                path: {
                    dataset_id,
                },
            },
        }
    );

    const deleteEpisodesMutation = $api.useMutation('delete', '/api/dataset/{dataset_id}/episodes', {
        onSuccess: (data) => {
            const query_key = [
                'get',
                '/api/dataset/{dataset_id}/episodes',
                {
                    params: {
                        path: {
                            dataset_id,
                        },
                    },
                },
            ];
            queryClient.setQueryData(query_key, data);
            setSelectedEpisodes([]);
        },
    });

    const deleteEpisodes = (episodeIndices: number[]) => {
        deleteEpisodesMutation.mutate({
            params: {
                path: {
                    dataset_id,
                },
            },
            body: episodeIndices,
        });
    };

    const isPending = deleteEpisodesMutation.isPending || episodesPending || datasetPending;

    return (
        <DatasetContext.Provider
            value={{
                dataset_id,
                dataset,
                deleteEpisodes,
                episodes,
                setSelectedEpisodes,
                selectedEpisodes,
                isPending,
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
