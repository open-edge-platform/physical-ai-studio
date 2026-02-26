import { useRef, useState } from 'react';

import { useMutation, useQueryClient } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import useWebSocketWithResponse from '../../../components/websockets/use-websocket-with-response';

interface TeleoperationState {
    initialized: boolean;
    is_recording: boolean;
    error: boolean;
}

interface RecordApiJsonResponse<Object> {
    event: string;
    data: Object;
}

function createTeleoperationState(data: unknown | null = null) {
    if (data) {
        return data as TeleoperationState;
    }
    return {
        initialized: false,
        is_recording: false,
        error: false,
    };
}

export interface Observation {
    timestamp: number;
    actions: { [key: string]: number };
    cameras: { [key: string]: string };
}

export const useTeleoperation = (setup: SchemaTeleoperationConfig, onError: (error: string) => void) => {
    const client = useQueryClient();
    const [state, setState] = useState<TeleoperationState>(createTeleoperationState());
    const { sendJsonMessage, readyState, sendJsonMessageAndWait } = useWebSocketWithResponse(
        fetchClient.PATH('/api/record/teleoperate/ws'),
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onOpen: () => init.mutate(),
            onClose: () => onClose(),
        }
    );

    const [numberOfRecordings, setNumberOfRecordings] = useState<number>(0);
    const observation = useRef<Observation | undefined>(undefined);

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as RecordApiJsonResponse<unknown>;
        switch (message.event) {
            case 'state':
                setState(message.data as TeleoperationState);
                break;
            case 'observations':
                observation.current = message['data'] as Observation;
                break;
            case 'episode':
                //onEpisode(message['data'] as SchemaEpisode);
                break;
            case 'error':
                onError(message['data'] as string);
        }
    };

    const invalidateEpisodesData = () => {
        if (setup.dataset.id) {
            const queryKey = [
                'get',
                '/api/dataset/{dataset_id}/episodes',
                {
                    params: {
                        path: {
                            dataset_id: setup.dataset.id,
                        },
                    },
                },
            ];

            client.invalidateQueries({ queryKey });
        }
    };

    const init = useMutation({
        mutationFn: async () =>
            await sendJsonMessageAndWait<RecordApiJsonResponse<TeleoperationState>>(
                { event: 'initialize', data: setup },
                (data) => data['data']['initialized'] == true
            ),
    });

    const startEpisode = () => {
        sendJsonMessage({
            event: 'start_recording',
            data: {},
        });
    };

    const onClose = () => {
        invalidateEpisodesData();
        setState(createTeleoperationState());
    };

    const saveEpisode = useMutation({
        mutationFn: async () => {
            const message = await sendJsonMessageAndWait<RecordApiJsonResponse<TeleoperationState>>(
                { event: 'save', data: {} },
                ({ data }) => data['is_recording'] == false
            );
            setNumberOfRecordings((n) => n + 1);
            return message;
        },
    });

    const cancelEpisode = () => {
        sendJsonMessage({
            event: 'cancel',
            data: {},
        });
    };

    const disconnect = () => {
        sendJsonMessage({
            event: 'disconnect',
            data: {},
        });
    };

    return {
        state,
        init,
        startEpisode,
        disconnect,
        saveEpisode,
        cancelEpisode,
        observation,
        readyState,
        numberOfRecordings,
    };
};
