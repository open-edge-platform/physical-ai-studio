import { Button } from '@geti/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { useCameraId } from '../use-camera';
import { useCameraFormBody } from './provider';

export const UpdateCameraButton = () => {
    const navigate = useNavigate();
    const { project_id, camera_id } = useCameraId();

    const updateCameraMutation = $api.useMutation('put', '/api/projects/{project_id}/cameras/{camera_id}');
    const body = useCameraFormBody(camera_id);

    return (
        <Button
            variant='accent'
            isPending={updateCameraMutation.isPending}
            isDisabled={body === null}
            onPress={async () => {
                if (body === null) {
                    return;
                }

                await updateCameraMutation.mutateAsync(
                    {
                        params: { path: { project_id, camera_id } },
                        body,
                    },
                    {
                        onSuccess: () => {
                            navigate(paths.project.cameras.show({ project_id, camera_id }));
                        },
                    }
                );
            }}
        >
            Update camera
        </Button>
    );
};
