import { useParams } from 'react-router';

import { $api } from '../../api/client';
import { SchemaEnvironmentWithRelations as SchemaEnvironment } from '../../api/openapi-spec';

export function useEnvironmentId() {
    const { environment_id, project_id } = useParams<{ environment_id: string; project_id: string }>();

    if (project_id === undefined || environment_id === undefined) {
        throw new Error('Unkown environment_id parameter');
    }

    return { project_id, environment_id };
}

export function useEnvironment(): SchemaEnvironment {
    const { project_id, environment_id } = useEnvironmentId();

    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: { path: { project_id, environment_id } },
        }
    );

    return environment;
}
