import { useEffect, useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';

import { API_BASE_URL } from '../../../api/client';
import useWebSocketWithResponse from '../../../components/websockets/use-websocket-with-response';
import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';

interface RobotState {
    initialized: boolean;
    is_recording: boolean;
    cameras: string[];
}

interface RecordApiJsonResponse {
    event: string;
    data: object;
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
  timestamp: number
  actions: { [key: string]: number }
  cameras: { [key: string]: string }
}

export const useRecording = (setup: SchemaTeleoperationConfig) => {
    const [state, setState] = useState<RobotState>(createRobotState());
    const { sendJsonMessage, lastJsonMessage, readyState, sendJsonMessageAndWait } = useWebSocketWithResponse(
      `${API_BASE_URL}/api/record/teleoperate/ws`,
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onOpen: () => init.mutate(),
            onClose: () => setState(createRobotState()),
        }
    );

    const [numberOfRecordings, setNumberOfRecordings] = useState<number>(0);
    const observation = useRef<Observation | undefined>(undefined);

    useEffect(() => {
        if (lastJsonMessage) {
            const message = lastJsonMessage as RecordApiJsonResponse;
            switch (message.event) {
                case 'state':
                    setState(message.data as RobotState);
                    break;
                case 'actions':
                    break;
            }
        }
    }, [lastJsonMessage]);

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as RecordApiJsonResponse;
        if (message['event'] === 'observations') {
            observation.current = message['data'] as Observation;
        }
    };

    const init = useMutation({
        mutationFn: async () => {
            await sendJsonMessageAndWait(
                { event: 'initialize', data: setup },
                ({data}) => JSON.parse(data)['data']['initialized'] == true
            );
        },
    });

    const startRecording = () => {
        sendJsonMessage({
            event: 'start_recording',
            data: {},
        });
    };

    const saveEpisode = useMutation({
        mutationFn: async () => {
            await sendJsonMessageAndWait(
                { event: 'save', data: {} },
                ({data}) => JSON.parse(data)['data']['is_recording'] == false
            );
            setNumberOfRecordings((n) => n + 1);
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
        startRecording,
        disconnect,
        saveEpisode,
        cancelEpisode,
        observation,
        readyState,
        numberOfRecordings,
    };
};
