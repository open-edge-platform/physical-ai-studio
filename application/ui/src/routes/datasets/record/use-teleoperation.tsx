import { useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';

import { fetchClient } from '../../../api/client';
import { SchemaEpisode, SchemaTeleoperationConfig } from '../../../api/openapi-spec';
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

export const useTeleoperation = (
    setup: SchemaTeleoperationConfig,
    onEpisode: (episode: SchemaEpisode) => void,
    onError: (error: string) => void
) => {
    const [state, setState] = useState<TeleoperationState>(createTeleoperationState());
    const { sendJsonMessage, readyState, sendJsonMessageAndWait } = useWebSocketWithResponse(
        fetchClient.PATH('/api/record/teleoperate/ws'),
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onOpen: () => init.mutate(),
            onClose: () => setState(createTeleoperationState()),
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
                onEpisode(message['data'] as SchemaEpisode);
                break;
            case 'error':
                onError(message['data'] as string);
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
