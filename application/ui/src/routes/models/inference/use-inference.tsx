import { useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { SchemaInferenceConfig } from '../../../api/openapi-spec';
import useWebSocketWithResponse from '../../../components/websockets/use-websocket-with-response';

interface InferenceState {
    initialized: boolean;
    is_running: boolean;
    task_index: number;
    error: boolean;
}

interface InferenceApiJsonResponse<T> {
    event: string;
    data: T;
}

export interface Observation {
    timestamp: number;
    state: { [joint: string]: number }; // robot joint state before inference
    actions: { [joint: string]: number }; // joint actions suggested by inference
    cameras: { [key: string]: string };
}

const createInferenceState = (): InferenceState => {
    return {
        initialized: false,
        is_running: false,
        task_index: 0,
        error: false,
    };
};

export const useInference = (setup: SchemaInferenceConfig, onError: (error: string) => void) => {
    const [state, setState] = useState<InferenceState>(createInferenceState());
    const observation = useRef<Observation | undefined>(undefined);

    const { sendJsonMessage, sendJsonMessageAndWait } = useWebSocketWithResponse(
        fetchClient.PATH('/api/record/inference/ws'),
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onOpen: () => init.mutate(),
            onClose: () => setState(createInferenceState()),
            onError: console.error,
        }
    );

    const init = useMutation({
        mutationFn: async () =>
            await sendJsonMessageAndWait<InferenceApiJsonResponse<InferenceState>>(
                { event: 'initialize', data: setup },
                (data) => data['data']['initialized'] == true
            ),
    });

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

    const startTask = (taskIndex: number) => {
        sendJsonMessage({
            event: 'start_task',
            data: { task_index: taskIndex },
        });
    };

    const disconnect = () => {
        sendJsonMessage({
            event: 'disconnect',
            data: {},
        });
    };
    const stop = () => {
        sendJsonMessage({
            event: 'stop',
            data: {},
        });
    };

    return {
        state,
        startTask,
        stop,
        disconnect,
        observation,
    };
};
