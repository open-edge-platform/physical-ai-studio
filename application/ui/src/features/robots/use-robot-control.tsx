import { useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import { SchemaDatasetOutput, SchemaEnvironmentWithRelations, SchemaModel } from '../../api/openapi-spec';
import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';

type FollowerSource = 'teleoperator' | 'model' | null;

interface RobotControlState {
    model_loaded: boolean;
    environment_loaded: boolean;
    task_index: number;
    error: boolean;
    is_recording: boolean;
    dataset_loaded: boolean;
    follower_source: FollowerSource;
}

const createRobotControlState = (): RobotControlState => {
    return {
        model_loaded: false,
        environment_loaded: false,
        dataset_loaded: false,
        task_index: 0,
        error: false,
        is_recording: false,
        follower_source: null,
    };
};

interface RobotControlApiJsonResponse<T> {
    event: string;
    data: T;
}
export interface Observation {
    timestamp: number;
    state: { [joint: string]: number }; // robot joint state before inference
    actions: { [joint: string]: number } | null; // joint actions suggested by inference
    cameras: { [key: string]: string };
}

interface useRobotControlProps {
    environment: SchemaEnvironmentWithRelations;
    model?: SchemaModel;
    dataset?: SchemaDatasetOutput;
    backend?: string;
    onError: (error: string) => void;
}
export const useRobotControl = ({ environment, model, dataset, backend, onError }: useRobotControlProps) => {
    const [state, setState] = useState<RobotControlState>(createRobotControlState());
    const observation = useRef<Observation | undefined>(undefined);

    const onOpen = () => {
        loadEnvironment.mutate(environment);
        if (model && backend) {
            loadModel.mutate({ model, backend });
            setFollowerSource.mutate('model');
        }
        if (dataset) {
            loadDataset.mutate(dataset);
            setFollowerSource.mutate('teleoperator');
        }
    };

    const { sendJsonMessageAndWait, readyState } = useWebSocketWithResponse(
        fetchClient.PATH('/api/record/robot_control/ws'),
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onError: console.error,
            onClose: () => setState(createRobotControlState()),
            onOpen,
        }
    );

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as RobotControlApiJsonResponse<unknown>;
        if (message['event'] === 'observations') {
            observation.current = message['data'] as Observation;
        }
        if (message['event'] === 'state') {
            setState(message['data'] as RobotControlState);
        }

        if (message['event'] === 'error') {
            onError(message['data'] as string);
        }
    };

    const loadDataset = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (datasetConfig: SchemaDatasetOutput) =>
            await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'load_dataset', data: { dataset: datasetConfig } },
                (data) => data['data']['dataset_loaded']
            ),
    });

    const loadEnvironment = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (env: SchemaEnvironmentWithRelations) =>
            await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'load_environment', data: { environment: env } },
                (data) => data['data']['environment_loaded']
            ),
    });

    const loadModel = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (properties: { model: SchemaModel; backend: string }) =>
            await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'load_model', data: properties },
                ({ data }) => data['model_loaded']
            ),
    });

    const startTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (task: string) =>
            await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'start_task', data: { task } },
                ({ data }) => data['follower_source'] === 'model'
            ),
    });

    const stopTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () =>
            await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'stop_task', data: {} },
                ({ data }) => data['follower_source'] === null
            ),
    });

    const startEpisode = useMutation({
        mutationFn: async (task: string) => {
            const message = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'start_recording', data: { task } },
                ({ data }) => data['is_recording'] == true
            );
            return message;
        },
    });

    const saveEpisode = useMutation({
        mutationFn: async () => {
            const message = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'save_episode', data: {} },
                ({ data }) => data['is_recording'] == false
            );
            return message;
        },
    });

    const discardEpisode = useMutation({
        mutationFn: async () => {
            const message = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'discard_episode', data: {} },
                ({ data }) => data['is_recording'] == false
            );
            return message;
        },
    });

    const setFollowerSource = useMutation({
        mutationFn: async (follower_source: FollowerSource) => {
            const message = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'set_follower_source', data: { follower_source } },
                ({ data }) => data['follower_source'] == follower_source
            );
            return message;
        },
    });

    const readyForInference = state.environment_loaded && state.model_loaded;
    const readyForRecording = state.environment_loaded && state.dataset_loaded;
    const isConnected = readyState === 1;

    return {
        observation,
        state,
        loadEnvironment,
        loadModel,
        loadDataset,
        startTask,
        stopTask,
        readyForInference,
        readyForRecording,
        setFollowerSource,
        startEpisode,
        saveEpisode,
        discardEpisode,
        isConnected,
    };
};
