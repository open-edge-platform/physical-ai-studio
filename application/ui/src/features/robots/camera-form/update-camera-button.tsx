import { Button, Flex } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { isRecordingLockedError } from '../../../api/errors';
import { InlineAlert } from '../../../features/robots/setup-wizard/shared/inline-alert';
import { paths } from '../../../router';
import { useCameraId } from '../use-camera';
import { useCameraFormBody } from './provider';

export const UpdateCameraButton = () => {
    const navigate = useNavigate();
    const { project_id, camera_id } = useCameraId();

    const updateCameraMutation = $api.useMutation('put', '/api/projects/{project_id}/cameras/{camera_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/cameras', { params: { path: { project_id } } }],
                [
                    'get',
                    '/api/projects/{project_id}/cameras/{camera_id}',
                    { params: { path: { project_id, camera_id } } },
                ],
            ],
        },
    });
    const body = useCameraFormBody(camera_id);

    const isLocked = isRecordingLockedError(updateCameraMutation.error);

    return (
        <Flex direction='column' gap='size-200'>
            {isLocked && (
                <InlineAlert variant='warning'>
                    Camera settings cannot be changed while a recording session is active.
                </InlineAlert>
            )}

            <Button
                variant='accent'
                isPending={updateCameraMutation.isPending}
                isDisabled={body === null}
                onPress={async () => {
                    if (body === null) {
                        return;
                    }

                    updateCameraMutation.reset();

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
        </Flex>
    );
};
