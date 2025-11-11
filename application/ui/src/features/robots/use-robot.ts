import { useParams } from 'react-router';

import { $api } from '../../api/client';
import { SchemaRobot } from '../../api/openapi-spec';

export function useRobotId() {
    const { robot_id, project_id } = useParams<{ robot_id: string; project_id: string }>();

    if (project_id === undefined || robot_id === undefined) {
        throw new Error('Unkown robot_id parameter');
    }

    return { project_id, robot_id };
}

export function useRobot(): SchemaRobot {
    const { project_id, robot_id } = useRobotId();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id } },
    });

    return robot;
}
