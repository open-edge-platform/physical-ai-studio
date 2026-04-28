import { useCallback, useEffect, useState } from 'react';

import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { mapJointToURDFJoint, urdfPathForType, useRobotModels } from './robot-models-context';
import { SchemaRobotType } from './robot-types';

type Joint = {
    name: string;
    value: number;
};
type JointsState = Array<Joint>;

type StateWasUpdatedEvent = {
    name: 'state_was_updated';
    is_controlled: boolean;
    // [joint_name]: robot state in degrees
    state: Record<string, number>;
};

const getNewJointState = (newJoints: StateWasUpdatedEvent['state']) => {
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

export const useJointState = (project_id: string, robot_id: string) => {
    const [joints, setJoints] = useState<JointsState>([]);

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const payload = JSON.parse(event.data);

            if (payload['event'] === 'state_was_updated') {
                const newJoints = getNewJointState(payload['state']);
                setJoints(newJoints);
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }, []);

    const socket = useWebSocket(
        fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}/ws', {
            params: { path: { project_id, robot_id } },
        }),
        {
            queryParams: {
                fps: 30,
            },
            share: true,
            shouldReconnect: () => true,
            reconnectAttempts: 5,
            reconnectInterval: 3000,
            onMessage: handleMessage,
            onError: (error) => console.error('WebSocket error:', error),
            onClose: () => console.info('WebSocket closed'),
        }
    );

    return { joints, socket };
};
