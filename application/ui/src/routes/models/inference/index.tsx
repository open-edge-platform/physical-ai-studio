import { ToastQueue } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getDefaultInferenceDevice, getSupportedInferenceDevices } from '../../../features/models/backend-selection';
import { RobotControlProvider } from '../../../features/robots/robot-control-provider';
import { InferenceViewer } from './inference-viewer';
import { useInferenceParams } from './use-inference-params';

export const Index = () => {
    const { project_id, model_id, backend, device } = useInferenceParams();

    const {
        data: { model },
    } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id } },
    });

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: { path: { dataset_id: model.dataset_id! } },
    });

    const { data: initialEnvironment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: { path: { project_id, environment_id: dataset.environment_id } },
        }
    );

    const { data: tasks } = $api.useSuspenseQuery('get', '/api/models/{model_id}/tasks', {
        params: { path: { model_id } },
    });

    const { data: inferenceDevices } = $api.useSuspenseQuery('get', '/api/system/devices/inference');
    const selectedDevice = getSupportedInferenceDevices(inferenceDevices, backend).find(
        (inferenceDevice) => inferenceDevice.device === device
    );
    const inferenceDevice = selectedDevice ?? getDefaultInferenceDevice(inferenceDevices, backend);

    return (
        <RobotControlProvider
            environment={initialEnvironment}
            model={model}
            inferenceDevice={inferenceDevice}
            onError={ToastQueue.negative}
        >
            <InferenceViewer tasks={tasks} />
        </RobotControlProvider>
    );
};
