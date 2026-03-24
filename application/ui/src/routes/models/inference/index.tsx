import { ToastQueue } from '@adobe/react-spectrum';

import { $api } from '../../../api/client';
import { RobotControlProvider } from '../../../features/robots/robot-control-provider';
import { InferenceViewer } from './inference-viewer';
import { useInferenceParams } from './use-inference-params';

export const Index = () => {
    const { project_id, model_id, backend } = useInferenceParams();

    const { data: model } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { query: { uuid: model_id } },
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
        params: { query: { uuid: model_id } },
    });

    return (
        <RobotControlProvider
            environment={initialEnvironment}
            model={model}
            backend={backend}
            onError={ToastQueue.negative}
        >
            <InferenceViewer tasks={tasks} />
        </RobotControlProvider>
    );
};
