import { useCallback, useRef, useState } from 'react';

import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';

interface JointState {
    name: string;
    value: number;
}

export type JointsState = Array<JointState>;

const getNewJointState = (newJoints: Record<string, number>): JointsState => {
    return Object.keys(newJoints).map((joint_name) => {
        return {
            name: joint_name,
            value: Number(newJoints[joint_name]),
        };
    });
};

export type ParsedRobotObservationMessage =
    | { type: 'observation'; joints: JointsState }
    | { type: 'state' }
    | { type: 'error'; message: string; errorCode: string }
    | { type: 'ignored' };

export const parseRobotObservationMessage = (payload: unknown): ParsedRobotObservationMessage => {
    if (typeof payload !== 'object' || payload === null || !('event' in payload)) {
        return { type: 'ignored' };
    }

    const message = payload as { event?: unknown; data?: unknown; message?: unknown; error_code?: unknown };

    if (message.event === 'observation') {
        if (typeof message.data !== 'object' || message.data === null) {
            return { type: 'ignored' };
        }
        return { type: 'observation', joints: getNewJointState(message.data as Record<string, number>) };
    }

    if (message.event === 'state') {
        return { type: 'state' };
    }

    if (message.event === 'error') {
        return {
            type: 'error',
            message: typeof message.message === 'string' ? message.message : 'Failed to connect to the robot.',
            errorCode: typeof message.error_code === 'string' ? message.error_code : 'robot_connection_failed',
        };
    }

    return { type: 'ignored' };
};

// Compose from the typed robot path; the observations socket is not in OpenAPI yet.
const observationSocketUrl = (project_id: string, robot_id: string): string =>
    `${fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id } },
    })}/observations/ws`;

export const useRobotObservations = (project_id: string, robot_id: string) => {
    const [joints, setJoints] = useState<JointsState>([]);
    const [error, setError] = useState<string | null>(null);
    const [errorCode, setErrorCode] = useState<string | null>(null);
    const hasErrorFrame = useRef(false);

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const parsed = parseRobotObservationMessage(JSON.parse(event.data));
            if (parsed.type === 'observation') {
                setJoints(parsed.joints);
            } else if (parsed.type === 'state') {
                hasErrorFrame.current = false;
                setError(null);
                setErrorCode(null);
            } else if (parsed.type === 'error') {
                hasErrorFrame.current = true;
                setError(parsed.message);
                setErrorCode(parsed.errorCode);
            }
        } catch (parseError) {
            console.error('Failed to parse WebSocket message:', parseError);
        }
    }, []);

    useWebSocket(observationSocketUrl(project_id, robot_id), {
        queryParams: {
            fps: 30,
        },
        share: true,
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (wsError) => console.error('WebSocket error:', wsError),
        onClose: (event) => {
            if (event.code !== 1000 && !hasErrorFrame.current) {
                setError(`Connection closed unexpectedly (code ${event.code})`);
                setErrorCode('connection_closed');
            }
        },
    });

    return { joints, error, errorCode };
};
