import { $api } from '../../../api/client';

export const useRecordingEnvironmentsQuery = (datasetId: string | null | undefined) =>
    $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}/recording-environments',
        { params: { path: { dataset_id: datasetId ?? '' } } },
        { enabled: datasetId != null }
    );

export const useDatasetEnvironmentQuery = (projectId: string, datasetId: string | null | undefined) => {
    const associations = useRecordingEnvironmentsQuery(datasetId);
    const environmentId = associations.data?.[0]?.environment_id;

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        { params: { path: { project_id: projectId, environment_id: environmentId ?? '' } } },
        { enabled: environmentId !== undefined }
    );
};
