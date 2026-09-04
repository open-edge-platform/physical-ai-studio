import { $api } from '../../../api/client';

export const useDatasetQuery = (datasetId: string | undefined | null) => {
    return $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        {
            params: { path: { dataset_id: String(datasetId) } },
        },
        {
            enabled: datasetId != null,
        }
    );
};

export const useEnvironmentQuery = (projectId: string, environmentId: string | undefined) => {
    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: {
                path: {
                    environment_id: String(environmentId),
                    project_id: projectId,
                },
            },
        },
        {
            enabled: environmentId !== undefined,
        }
    );
};
