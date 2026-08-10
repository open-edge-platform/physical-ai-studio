import { useCallback, useEffect, useRef, useState } from 'react';

import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { useRobotCatalogDefinitionQuery } from './robot-catalog.hooks';
import { mapJointToURDFJoint, useLoadModelQuery } from './robot-models-context';
import { SchemaRobotType } from './robot-types';

type JointsState = Array<{
    name: string;
    value: number;
}>;

const getNewJointState = (newJoints: Record<string, number>) => {
    return Object.keys(newJoints).map((joint_name) => {
        return {
            name: joint_name,
            value: Number(newJoints[joint_name]),
        };
    });
};

export const useSynchronizeModelJoints = (joints: JointsState, robotType: SchemaRobotType) => {
    const { data: definition } = useRobotCatalogDefinitionQuery(robotType);
    const jointMap = definition.joint_map;

    const { data: model } = useLoadModelQuery(robotType);

    useEffect(() => {
        if (!model) return;

        joints.forEach((joint) => {
            mapJointToURDFJoint(joint, model, jointMap);
        });
    }, [model, joints, jointMap]);
};

// Mirrors the backend's runtime.contract.FollowerSource. 'policy' is not sent by
// this websocket yet (reserved for Phase B inference), but the union must match
// the backend contract so a future value doesn't silently narrow to 'hold'.
export type FollowerSource = 'hold' | 'teleop' | 'policy';

const RECOVERABLE_ERROR_CODES = new Set(['leader_connection_lost']);

export const isRecoverableRobotControlError = (errorCode: unknown): errorCode is string =>
    typeof errorCode === 'string' && RECOVERABLE_ERROR_CODES.has(errorCode);

interface RobotControlState {
    connected: boolean;
    follower_source: FollowerSource;
}

export const useJointState = (project_id: string, follower_id: string, leader_id?: string) => {
    const [joints, setJoints] = useState<JointsState>([]);
    const [state, setState] = useState<RobotControlState>({
        connected: false,
        follower_source: 'hold',
    });
    const [error, setError] = useState<string | null>(null);
    const [errorCode, setErrorCode] = useState<string | null>(null);
    const [warning, setWarning] = useState<string | null>(null);
    const hasFatalError = useRef(false);

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const payload = JSON.parse(event.data);

            if (payload['event'] === 'observation') {
                const newJoints = getNewJointState(payload['data']);
                setJoints(newJoints);
            } else if (payload['event'] === 'state') {
                setState(payload['data']);
                setError(null);
                setErrorCode(null);
                setWarning(null);
                hasFatalError.current = false;
            } else if (payload['event'] === 'error') {
                if (isRecoverableRobotControlError(payload.error_code)) {
                    setWarning(
                        typeof payload.message === 'string'
                            ? payload.message
                            : 'The robot session entered a safe state.'
                    );
                    return;
                }
                hasFatalError.current = true;
                setError(typeof payload.message === 'string' ? payload.message : 'Failed to connect to the robot.');
                setErrorCode(typeof payload.error_code === 'string' ? payload.error_code : 'robot_connection_failed');
            }
        } catch (parseError) {
            console.error('Failed to parse WebSocket message:', parseError);
        }
    }, []);

    const socket = useWebSocket(
        fetchClient.PATH('/api/projects/{project_id}/robots/ws', {
            params: { path: { project_id } },
        }),
        {
            queryParams: {
                fps: 30,
            },
            share: true,
            shouldReconnect: () => !hasFatalError.current,
            reconnectAttempts: 5,
            reconnectInterval: 3000,
            onOpen: () => {
                if (hasFatalError.current) {
                    return;
                }
                setError(null);
                setErrorCode(null);
                setWarning(null);
                socket.sendJsonMessage({
                    follower_id,
                    leader_id,
                });
            },
            onMessage: handleMessage,
            onError: (wsError) => console.error('WebSocket error:', wsError),
            onClose: (event) => {
                if (!hasFatalError.current && event.code !== 1000) {
                    hasFatalError.current = true;
                    setError(`Connection closed unexpectedly (code ${event.code})`);
                    setErrorCode('connection_closed');
                }
            },
        }
    );

    const setFollowerSourceRequest = (value: FollowerSource) => {
        socket.sendJsonMessage({
            event: 'set_follower_source',
            data: { follower_source: value },
        });
    };

    return { joints, socket, state, error, errorCode, warning, setFollowerSource: setFollowerSourceRequest };
};
