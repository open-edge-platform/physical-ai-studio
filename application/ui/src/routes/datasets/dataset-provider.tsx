import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { $api } from '../../api/client';
import { SchemaDatasetOutput, SchemaEpisode } from '../../api/openapi-spec';

type DatasetContextValue = null | {
    dataset_id: string;
    dataset: SchemaDatasetOutput;
    episodes: SchemaEpisode[];
    setSelectedEpisodes: Dispatch<SetStateAction<number[]>>;
    selectedEpisodes: number[];
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

    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

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
