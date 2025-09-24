import { useParams } from 'react-router';

import { $api } from '../../api/client';
import { SchemaProjectInput } from '../../api/openapi-spec';

export function useProjectId(): { project_id: string } {
    const { project_id } = useParams<{ project_id: string }>();

    if (project_id === undefined) {
        throw new Error('Unkown project_id parameter');
    }

    return { project_id };
}

export function useProject(): SchemaProjectInput {
    const { project_id } = useProjectId();

    const { data: project } = $api.useSuspenseQuery('get', '/api/projects/{id}', {
        params: { path: { id: project_id } },
    });

    return project;
}
