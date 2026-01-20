import { useParams } from 'react-router';

import { $api } from '../../api/client';
import { SchemaProjectCamera } from '../../api/types';

export function useCameraId() {
    const { camera_id, project_id } = useParams<{ camera_id: string; project_id: string }>();

    if (project_id === undefined || camera_id === undefined) {
        throw new Error('Unkown camera_id parameter');
    }

    return { project_id, camera_id };
}

export function useCamera(): SchemaProjectCamera {
    const { project_id, camera_id } = useCameraId();

    const { data: camera } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras/{camera_id}', {
        params: { path: { project_id, camera_id } },
    });

    return camera;
}
