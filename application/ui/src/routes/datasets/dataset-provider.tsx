import { createContext, ReactNode, useContext } from 'react';

import { SchemaDatasetOutput, SchemaEpisode } from '../../api/openapi-spec';
import { $api } from '../../api/client';
import { useQueryClient } from '@tanstack/react-query';

type DatasetContextValue = null | {
    dataset_id: string;
    dataset: SchemaDatasetOutput;
    episodes: SchemaEpisode[];
    deleteEpisodes: (episodeIndices: number[]) => void;
};
const DatasetContext = createContext<DatasetContextValue>(null);

interface DatasetProviderProps {
    dataset_id: string;
    children: ReactNode;
}
export const DatasetProvider = ({ dataset_id, children }: DatasetProviderProps) => {
    const queryClient = useQueryClient();

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: {
            path: {
                dataset_id
            }
        }
    })

    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

    const deleteEpisodesMutation = $api.useMutation('delete', '/api/dataset/{dataset_id}/episodes', {
        onSuccess: (data) => {
            const query_key = [
                "get",
                "/api/dataset/{dataset_id}/episodes",
                {
                    "params": {
                        "path": {
                            dataset_id
                        }
                    }
                }
            ]
            queryClient.setQueryData(query_key, data)
        }
    })

    const deleteEpisodes = (episodeIndices: number[]) => {
        deleteEpisodesMutation.mutate({
            params: {
                path: {
                    dataset_id
                }
            },
            body: episodeIndices
        })

    }

    return (
        <DatasetContext.Provider value={{
            dataset_id,
            dataset,
            deleteEpisodes,
            episodes,
        }}>
            {children}
        </DatasetContext.Provider>
    )
}

export const useDataset = () => {
    const ctx = useContext(DatasetContext);
    if (!ctx) throw new Error('useDataset must be used within DatasetProvider');
    return ctx;
}
