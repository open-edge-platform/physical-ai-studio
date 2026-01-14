import { useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';

import { SchemaEpisode, SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import useWebSocketWithResponse from '../../../components/websockets/use-websocket-with-response';

interface RobotState {
    initialized: boolean;
    is_recording: boolean;
    cameras: string[];
}

interface RecordApiJsonResponse<Object> {
    event: string;
    data: Object;
}

function createRobotState(data: unknown | null = null) {
    if (data) {
        return data as RobotState;
    }
    return {
        initialized: false,
        is_recording: false,
        cameras: [],
    };
}

export interface Observation {
    timestamp: number;
    actions: { [key: string]: number };
    cameras: { [key: string]: string };
}

export const useTeleoperation = (setup: SchemaTeleoperationConfig, onEpisode: (episode: SchemaEpisode) => void) => {
    const [state, setState] = useState<RobotState>(createRobotState());
    const { sendJsonMessage, readyState, sendJsonMessageAndWait } = useWebSocketWithResponse(
        `/api/record/teleoperate/ws`,
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onOpen: () => init.mutate(),
            onClose: () => setState(createRobotState()),
        }
    );

    const [numberOfRecordings, setNumberOfRecordings] = useState<number>(0);
    const observation = useRef<Observation | undefined>(undefined);

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as RecordApiJsonResponse<object>;
        switch (message.event) {
            case 'state':
                setState(message.data as RobotState);
                break;
            case 'observations':
                observation.current = message['data'] as Observation;
                break;
            case 'episode':
                onEpisode(message['data'] as SchemaEpisode);
                break;
        }
    };

    const init = useMutation({
        mutationFn: async () =>
            await sendJsonMessageAndWait<RecordApiJsonResponse<RobotState>>(
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
            const message = await sendJsonMessageAndWait<RecordApiJsonResponse<RobotState>>(
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
