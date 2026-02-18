import { useCallback, useEffect, useState } from 'react';

import { View } from '@geti/ui';
import useWebSocket from 'react-use-websocket';
import { degToRad } from 'three/src/math/MathUtils.js';

import { $api } from '../../../../api/client';
import { useProjectId } from '../../../projects/use-project';
import { RobotViewer } from '../../controller/robot-viewer';
import { RobotModelsProvider, useRobotModels } from '../../robot-models-context';

type JointsState = Array<{
    name: string;
    value: number;
}>;

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

const useSynchronizeModelJoints = (joints: JointsState) => {
    const { models } = useRobotModels();

    function removeSuffix(str: string, suffix: string): string {
        return str.endsWith(suffix) ? str.slice(0, -suffix.length) : str;
    }

    useEffect(() => {
        models.forEach((model) => {
            joints.forEach((joint) => {
                const name = removeSuffix(joint.name, '.pos');

                if (name === 'gripper' && model.robotName == 'wxai') {
                    model.setJointValue('left_carriage_joint', joint.value); // meters
                    return;
                }

                if (model.joints[name]) {
                    model.setJointValue(name, degToRad(joint.value));
                }
            });
        });
    }, [models, joints]);
};

const useJointState = (project_id: string, robot_id: string) => {
    const [joints, setJoints] = useState<JointsState>([]);

    useSynchronizeModelJoints(joints);

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

    useWebSocket(`/api/projects/${project_id}/robots/${robot_id}/ws`, {
        queryParams: {
            fps: 60,
        },
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: handleMessage,
        onError: (error) => console.error('WebSocket error:', error),
        onClose: () => console.info('WebSocket closed'),
    });
};

const InnerCell = ({ robot_id }: { robot_id: string }) => {
    const { project_id } = useProjectId();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id } },
    });

    useJointState(project_id, robot_id);

    return (
        <View minWidth='size-4000' minHeight='size-4000' width='100%' height='100%' backgroundColor={'gray-600'}>
            <RobotViewer robot={robot} />
        </View>
    );
};

export const RobotCell = ({ robot_id }: { robot_id: string }) => {
    return (
        <RobotModelsProvider>
            <InnerCell robot_id={robot_id} />
        </RobotModelsProvider>
    );
};
