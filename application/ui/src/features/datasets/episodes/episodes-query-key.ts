import { getQueryKey } from '../../../query-client/query-client.interface';

export const getEpisodesQueryKey = (dataset_id: string) =>
    getQueryKey(['get', '/api/dataset/{dataset_id}/episodes', { params: { path: { dataset_id } } }]);
