import { $api } from '../../api/client';

export function useDataset(project_id: string, repo_id: string) {
    const [repo, id] = repo_id.split('/');
    const { data: dataset } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/datasets/{repo}/{id}', {
        params: { path: { project_id, repo, id } },
    });

    return {
        dataset,
    };
}
