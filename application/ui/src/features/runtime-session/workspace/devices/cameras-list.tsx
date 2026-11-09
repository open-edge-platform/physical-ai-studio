import { $api } from '../../../../api/client';
import { SchemaSessionCamera, SchemaStatusMessage } from '../../../../api/openapi-spec';
import { useProjectId } from '../../../projects/use-project';
import { CameraCard } from './camera-card';

interface CamerasListProps {
    cameras: { camera: SchemaSessionCamera; status: SchemaStatusMessage }[];
}

export const CamerasList = ({ cameras }: CamerasListProps) => {
    const { project_id } = useProjectId();
    const { data: projectCameras } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    return (
        <>
            {cameras.map(({ camera, status }) => (
                <CameraCard
                    key={camera.key}
                    camera={camera}
                    projectCamera={projectCameras.find(({ id }) => id === camera.camera_id)}
                    status={status}
                />
            ))}
        </>
    );
};
