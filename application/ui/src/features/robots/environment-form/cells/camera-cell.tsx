import { $api } from '../../../../api/client';
import { WebsocketCamera } from '../../../cameras/websocket-camera';
import { useProjectId } from '../../../projects/use-project';

export const CameraCell = ({ camera_id }: { camera_id: string }) => {
    const { project_id } = useProjectId();
    const cameraQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras/{camera_id}', {
        params: { path: { project_id, camera_id } },
    });

    return <WebsocketCamera camera={cameraQuery.data} />;
};
