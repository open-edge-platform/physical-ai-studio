import { Button } from '@geti/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useCameraFormBody } from './provider';

export const SubmitNewCameraButton = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();

    const addCameraMutation = $api.useMutation('post', '/api/projects/{project_id}/cameras');

    const body = useCameraFormBody(uuidv4());

    return (
        <Button
            variant='accent'
            isPending={addCameraMutation.isPending}
            isDisabled={body === null}
            onPress={async () => {
                if (body === null) {
                    return;
                }

                await addCameraMutation.mutateAsync(
                    {
                        params: { path: { project_id } },
                        body,
                    },
                    {
                        onSuccess: ({}, { body: { id: camera_id } }) => {
                            navigate(paths.project.cameras.show({ project_id, camera_id }));
                        },
                    }
                );
            }}
        >
            Add camera
        </Button>
    );
};
