import { useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';

import { API_BASE_URL } from '../../../api/client';
import { SchemaInferenceConfig } from '../../../api/openapi-spec';
import useWebSocketWithResponse from '../../../components/websockets/use-websocket-with-response';

interface InferenceState {
    initialized: boolean;
    is_running: boolean;
    task_index: number;
}

interface RecordApiJsonResponse {
    event: string;
    data: object;
}

export interface Observation {
    timestamp: number;
    state: { [key: string]: number }; // robot state
    actions: { [key: string]: number }; // actions suggested by inference
    cameras: { [key: string]: string };
}


const createInferenceState = (): InferenceState => {
    return {
        initialized: false,
        is_running: false,
        task_index: 0,
    };

}

export const useInference = (setup: SchemaInferenceConfig) => {
    const [state, setState] = useState<InferenceState>(createInferenceState());
    const observation = useRef<Observation | undefined>(undefined);

    const { sendJsonMessage, lastJsonMessage, readyState, sendJsonMessageAndWait } = useWebSocketWithResponse(
        `${API_BASE_URL}/api/record/inference/ws`,
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onOpen: () => init.mutate(),
            onClose: () => setState(createInferenceState()),
            onError: console.error,
        }
    );

    const init = useMutation({
        mutationFn: async () => await sendJsonMessageAndWait(
                { event: 'initialize', data: setup },
                (data) => data['data']['initialized'] == true
        )
    });

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as RecordApiJsonResponse;
        if (message['event'] === 'observations') {
            observation.current = message['data'] as Observation;
        }
        if (message['event'] === 'state') {
            setState(message['data'] as InferenceState);
        }
    };

    const startTask = (taskIndex: number) => {
        sendJsonMessage({
            event: 'start_task',
            data: { task_index: taskIndex },
        });
    };

    const calculateTrajectory = useMutation({
        mutationFn: async () => {
            const {data} = await sendJsonMessageAndWait(
                { event: 'calculate_trajectory', data: {} },
                (data) => data['event'] === "trajectory"
            );
            return data["trajectory"];
        }
    });

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
        calculateTrajectory,
    }
}
