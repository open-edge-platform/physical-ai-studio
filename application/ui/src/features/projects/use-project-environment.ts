import { $api } from '../../api/client';
import { SchemaEnvironmentWithRelations } from '../../api/openapi-spec';
import { useProjectId } from './use-project';

export function useProjectEnvironment(): SchemaEnvironmentWithRelations {
    const { project_id } = useProjectId();
    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: {
            path: {
                project_id,
            },
        },
    });
    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: {
                path: {
                    project_id,
                    environment_id: environments[0].id,
                },
            },
        }
    );

    return environment;
}
