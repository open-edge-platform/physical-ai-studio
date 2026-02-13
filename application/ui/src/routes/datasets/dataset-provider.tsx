import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { $api } from '../../api/client';
import { SchemaDatasetOutput, SchemaEpisode } from '../../api/openapi-spec';
import { useDeleteEpisodeQuery } from '../../features/datasets/episodes/use-episodes';

type DatasetContextValue = null | {
    dataset_id: string;
    dataset: SchemaDatasetOutput;
    episodes: SchemaEpisode[];
    deleteSelectedEpisodes: () => void;
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
    const [selectedEpisodes, setSelectedEpisodes] = useState<number[]>([]);

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
        }
    );

    const deleteEpisodesMutation = useDeleteEpisodeQuery();

    const deleteSelectedEpisodes = () => {
        deleteEpisodesMutation.mutate({
            params: {
                path: {
                    dataset_id,
                },
            },
            body: selectedEpisodes,
        });
        setSelectedEpisodes([]);
    };

    const isPending = deleteEpisodesMutation.isPending

    return (
        <DatasetContext.Provider
            value={{
                dataset_id,
                dataset,
                deleteSelectedEpisodes,
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
