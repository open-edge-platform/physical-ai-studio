import { useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { SchemaDatasetOutput, SchemaEnvironmentWithRelations, SchemaModel } from '../../../api/openapi-spec';
import useWebSocketWithResponse from '../../../components/websockets/use-websocket-with-response';

interface InferenceState {
    model_loaded: boolean;
    environment_loaded: boolean;
    is_running: boolean;
    task_index: number;
    error: boolean;
    is_recording: boolean;
    dataset_loaded: boolean;
}

const createInferenceState = (): InferenceState => {
    return {
        model_loaded: false,
        environment_loaded: false,
        dataset_loaded: false,
        is_running: false,
        task_index: 0,
        error: false,
        is_recording: false,
    };
};

interface InferenceApiJsonResponse<T> {
    event: string;
    data: T;
}
export interface Observation {
    timestamp: number;
    state: { [joint: string]: number }; // robot joint state before inference
    actions: { [joint: string]: number } | null; // joint actions suggested by inference
    cameras: { [key: string]: string };
}

interface useInferenceProps {
    environment: SchemaEnvironmentWithRelations;
    model?: SchemaModel;
    dataset?: SchemaDatasetOutput;
    backend?: string;
    onError: (error: string) => void;
}
export const useInference = ({environment, model, dataset, backend, onError}: useInferenceProps) => {
    const [state, setState] = useState<InferenceState>(createInferenceState());
    const observation = useRef<Observation | undefined>(undefined);

    const onOpen = () => {
        loadEnvironment.mutate(environment);
        if (model && backend) {
            loadModel.mutate({ model, backend });
        }
        if (dataset) {
            loadDataset.mutate(dataset);
        }
    };

    const { sendJsonMessageAndWait, readyState } = useWebSocketWithResponse(
        fetchClient.PATH('/api/record/inference/ws'),
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onError: console.error,
            onClose: () => setState(createInferenceState()),
            onOpen,
        }
    );

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as InferenceApiJsonResponse<unknown>;
        if (message['event'] === 'observations') {
            observation.current = message['data'] as Observation;
        }
        if (message['event'] === 'state') {
            setState(message['data'] as InferenceState);
        }

        if (message['event'] === 'error') {
            onError(message['data'] as string);
        }
    };

    const loadDataset = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (dataset: SchemaDatasetOutput) =>
            await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'load_dataset', data: { dataset } },
                (data) => data['data']['dataset_loaded']
            ),
    });

    const loadEnvironment = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (env: SchemaEnvironmentWithRelations) =>
            await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'load_environment', data: { environment: env } },
                (data) => data['data']['environment_loaded']
            ),
    });

    const loadModel = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (properties: { model: SchemaModel; backend: string }) =>
            await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'load_model', data: properties },
                ({ data }) => data['model_loaded']
            ),
    });

    const startTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (task: string) =>
            await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'start_task', data: { task }},
                ({ data }) => data['is_running']
            ),
    });

    const stopTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () =>
            await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'stop_task', data: {} },
                ({ data }) => data['is_running'] == false
            ),
    });

    const startEpisode = useMutation({
        mutationFn: async (task: string) => {
            const message = await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'start_recording', data: { task} },
                ({ data }) => data['is_recording'] == true
            );
            return message;
        },
    });

    const saveEpisode = useMutation({
        mutationFn: async () => {
            const message = await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'save_episode', data: {} },
                ({ data }) => data['is_recording'] == false
            );
            return message;
        },
    });

    const discardEpisode = useMutation({
        mutationFn: async () => {
            const message = await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'discard_episode', data: {} },
                ({ data }) => data['is_recording'] == false
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
        startEpisode,
        saveEpisode,
        discardEpisode,
        isConnected,
    };
};
