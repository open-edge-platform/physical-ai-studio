import { $api } from '../../../api/client';

export interface FinalizeFields {
    defaultTask: string;
    environmentId: string | undefined;
}

export const useDatasetImportJobQuery = (importJobId: string | undefined) => {
    return $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        {
            params: { path: { job_id: importJobId ?? '' } },
        },
        {
            enabled: importJobId !== undefined,
            refetchInterval: 1000,
            select: (job) => {
                return job.type === 'dataset_import' ? job : null;
            },
        }
    );
};
