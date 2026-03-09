import { useRef, useState } from 'react';

import { SchemaEnvironmentWithRelations, SchemaModel } from '../../../api/openapi-spec';
import { fetchClient } from '../../../api/client';
import useWebSocketWithResponse from '../../../components/websockets/use-websocket-with-response';
import { useMutation } from '@tanstack/react-query';


interface InferenceState {
    model_loaded: boolean;
    environment_loaded: boolean;
    is_running: boolean;
    task_index: number;
    error: boolean;
}

const createInferenceState = (): InferenceState => {
    return {
        model_loaded: false,
        environment_loaded: false,
        is_running: false,
        task_index: 0,
        error: false,
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

export const useInference2 = (onError: (error: string) => void) => {
    const [state, setState] = useState<InferenceState>(createInferenceState());
    const observation = useRef<Observation | undefined>(undefined);

    const { sendJsonMessage, sendJsonMessageAndWait, readyState } = useWebSocketWithResponse(
        fetchClient.PATH('/api/record/inference/ws'),
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onClose: () => setState(createInferenceState()),
            onError: console.error,
        }
    );

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as InferenceApiJsonResponse<unknown>;
        if (message['event'] === 'observations') {
            console.log(message)
            observation.current = message['data'] as Observation;
        }
        if (message['event'] === 'state') {
            setState(message['data'] as InferenceState);
        }

        if (message['event'] === 'error') {
            onError(message['data'] as string);
        }
    };

    const loadEnvironment = useMutation({
        mutationFn: async (environment: SchemaEnvironmentWithRelations) =>
            await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'load_environment', data: environment },
                (data) => data['data']['environment_loaded'] == true // TODO: Handle errors?
            ),
    });

    const loadModel = useMutation({
        mutationFn: async (data: { model: SchemaModel, backend: string }) =>
            await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'load_model', data },
                (data) => data['data']['model_loaded'] == true // TODO: Handle errors?
            ),
    });

    const startTask = (taskIndex: number) => {
        sendJsonMessage({
            event: 'start_task',
            data: { task_index: taskIndex },
        });
    };

    const stop = () => {
        sendJsonMessage({
            event: 'stop',
            data: {},
        });
    };

    const readyForInference = state.environment_loaded && state.model_loaded;
    const isConnected = readyState === 1;

    return {
        observation,
        state,
        loadEnvironment,
        loadModel,
        startTask,
        stop,
        readyForInference,
        isConnected,
    }

}
