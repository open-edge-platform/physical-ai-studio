import { useCallback, useEffect, useState } from 'react';

import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { mapJointToURDFJoint, useRobotModels } from './robot-models-context';
import { SchemaRobotType } from './robot-types';
import { urdfPathForType } from './robots-configuration';

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
    const { getModel } = useRobotModels();
    const urdfPath = urdfPathForType(robotType);
    const model = getModel(urdfPath);

    useEffect(() => {
        if (!model) return;

        joints.forEach((joint) => {
            mapJointToURDFJoint(joint, model, robotType);
        });
    }, [model, joints, robotType]);
};

export enum RobotActionReadState {
    None = 0,
    Teleoperation = 1,
    FromActions = 2,
}

interface RobotControlState {
    connected: boolean;
    follower_source: RobotActionReadState;
}

export const useJointState = (project_id: string, follower_id: string, leader_id?: string) => {
    const [joints, setJoints] = useState<JointsState>([]);
    const [state, setState] = useState<RobotControlState>({
        connected: false,
        follower_source: RobotActionReadState.None,
    });

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const payload = JSON.parse(event.data);

            if (payload['event'] === 'observation') {
                const newJoints = getNewJointState(payload['data']);
                setJoints(newJoints);
            } else if (payload['event'] === 'state') {
                setState(payload['data']);
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
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
            shouldReconnect: () => true,
            reconnectAttempts: 5,
            reconnectInterval: 3000,
            onOpen: () => {
                socket.sendJsonMessage({
                    follower_id,
                    leader_id,
                });
            },
            onMessage: handleMessage,
            onError: (error) => console.error('WebSocket error:', error),
            onClose: () => console.info('WebSocket closed'),
        }
    );

    const setFollowerSourceRequest = (value: RobotActionReadState) => {
        socket.sendJsonMessage({
            event: 'set_follower_source',
            data: value,
        });
    };

    return { joints, socket, state, setFollowerSource: setFollowerSourceRequest };
};
