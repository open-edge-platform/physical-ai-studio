import { createContext, ReactNode, useContext } from 'react';

import { $api } from '../../api/client';
import { SchemaDataset } from '../../api/openapi-spec';

interface DatasetContext {
    dataset: SchemaDataset;
}
export const DatasetContext = createContext<DatasetContext | undefined>(undefined);

interface DatasetProviderProps {
    children: ReactNode;
    project_id: string;
    repo_id: string;
}
export const DatasetProvider = ({ children, project_id, repo_id }: DatasetProviderProps) => {
    const [repo, id] = repo_id.split('/');
    const { data: dataset } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/datasets/{repo}/{id}', {
        params: { path: { project_id, repo, id } },
    });

    return (
        <DatasetContext.Provider
            value={{
                dataset,
            }}
        >
            {children}
        </DatasetContext.Provider>
    );
};

export function useDataset(): DatasetContext {
    const context = useContext(DatasetContext);

    if (context === undefined) {
        throw new Error('No dataset context');
    }

    return context;
}
