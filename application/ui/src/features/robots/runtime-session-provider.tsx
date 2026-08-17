import { createContext, ReactNode, RefObject, useContext, useRef, useState } from 'react';

import { useMutation, UseMutationResult } from '@tanstack/react-query';

import { SchemaEnvironmentWithRelations, SchemaInferenceDeviceInfo, SchemaModel } from '../../api/openapi-spec';
import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';
import { useProjectId } from '../projects/use-project';
import { FollowerSource, runtimeSocketUrl } from './use-joint-state';

type InferenceDevice = Pick<SchemaInferenceDeviceInfo, 'backend' | 'device'>;

interface RuntimeSessionState {
    connected: boolean;
    follower_source: FollowerSource;
    model_loaded: boolean;
    task: string | null;
}

const createRuntimeSessionState = (): RuntimeSessionState => ({
    connected: false,
    follower_source: 'hold',
    model_loaded: false,
    task: null,
});

interface RuntimeApiJsonResponse<T = RuntimeSessionState> {
    event: string;
    data?: T;
    message?: string;
    error_code?: string;
}

interface RuntimeSessionProviderProps {
    children: ReactNode;
    environment: SchemaEnvironmentWithRelations;
    model?: SchemaModel;
    inferenceDevice?: InferenceDevice;
    onError: (error: string) => void;
}

type MutationResult<TVariables = void> = UseMutationResult<RuntimeApiJsonResponse, Error, TVariables>;

type RuntimeSessionContextValue = {
    observation: RefObject<Record<string, number> | undefined>;
    environment: SchemaEnvironmentWithRelations;
    model: SchemaModel | undefined;
    inferenceDevice: InferenceDevice | undefined;
    state: RuntimeSessionState;
    loadModel: MutationResult<{ model: SchemaModel; inference_device: InferenceDevice }>;
    startTask: MutationResult<string>;
    stopTask: MutationResult;
    setFollowerSource: MutationResult<FollowerSource>;
    readyForInference: boolean;
    isConnected: boolean;
};

const RuntimeSessionContext = createContext<RuntimeSessionContextValue | null>(null);

const handshakeDevices = (environment: SchemaEnvironmentWithRelations) => {
    const follower = environment.robots?.[0];
    if (follower === undefined) {
        throw new Error('Cannot start a runtime session without a follower robot.');
    }
    const leader_id = follower.tele_operator.type === 'robot' ? follower.tele_operator.robot_id : undefined;
    return {
        follower_id: follower.robot.id,
        leader_id,
        camera_ids: (environment.cameras ?? []).flatMap((camera) => (camera.id ? [camera.id] : [])),
    };
};

export const RuntimeSessionProvider = (props: RuntimeSessionProviderProps) => {
    const { project_id } = useProjectId();
    const [state, setState] = useState<RuntimeSessionState>(createRuntimeSessionState());
    const observation = useRef<Record<string, number> | undefined>(undefined);
    const [model, setModel] = useState<SchemaModel | undefined>(props.model);
    const [inferenceDevice, setInferenceDevice] = useState<InferenceDevice | undefined>(props.inferenceDevice);
    const devices = handshakeDevices(props.environment);

    const onOpen = () => {
        socket.sendJsonMessage({
            follower_id: devices.follower_id,
            leader_id: devices.leader_id,
            camera_ids: devices.camera_ids,
        });
        if (props.model && props.inferenceDevice) {
            loadModel.mutate({ model: props.model, inference_device: props.inferenceDevice });
        }
    };

    const socket = useWebSocketWithResponse(runtimeSocketUrl(project_id, devices.follower_id), {
        shouldReconnect: () => true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onMessage: (event: WebSocketEventMap['message']) => {
            const message = JSON.parse(event.data) as RuntimeApiJsonResponse<unknown>;
            if (message.event === 'observation' && message.data !== undefined && typeof message.data === 'object') {
                observation.current = message.data as Record<string, number>;
            }
            if (message.event === 'state' && message.data !== undefined && typeof message.data === 'object') {
                const next = message.data as Partial<RuntimeSessionState>;
                setState({
                    connected: next.connected ?? false,
                    follower_source: next.follower_source ?? 'hold',
                    model_loaded: next.model_loaded ?? false,
                    task: next.task ?? null,
                });
            }
            if (message.event === 'error') {
                props.onError(typeof message.message === 'string' ? message.message : 'An unexpected error occurred.');
            }
        },
        onError: console.error,
        onClose: () => {
            setState(createRuntimeSessionState());
        },
        onOpen,
    });

    const loadModel = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (properties: { model: SchemaModel; inference_device: InferenceDevice }) => {
            const result = await socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                {
                    event: 'load_model',
                    data: {
                        model_id: properties.model.id,
                        inference_device: properties.inference_device,
                    },
                },
                ({ event, data }) => event === 'state' && data?.model_loaded === true
            );
            setModel(properties.model);
            setInferenceDevice(properties.inference_device);
            return result;
        },
    });

    const startTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (task: string) =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'start_task', data: { task } },
                ({ event, data }) => event === 'state' && data?.follower_source === 'policy'
            ),
    });

    const stopTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'stop_task', data: {} },
                ({ event, data }) => event === 'state' && data?.follower_source === 'hold'
            ),
    });

    const setFollowerSource = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (follower_source: FollowerSource) =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'set_follower_source', data: { follower_source } },
                ({ event, data }) => event === 'state' && data?.follower_source === follower_source
            ),
    });

    return (
        <RuntimeSessionContext.Provider
            value={{
                observation,
                environment: props.environment,
                model,
                inferenceDevice,
                state,
                loadModel,
                startTask,
                stopTask,
                setFollowerSource,
                readyForInference: state.connected && state.model_loaded,
                isConnected: socket.readyState === 1,
            }}
        >
            {props.children}
        </RuntimeSessionContext.Provider>
    );
};

export const useRuntimeSession = () => {
    const ctx = useContext(RuntimeSessionContext);
    if (!ctx) throw new Error('useRuntimeSession must be used within RuntimeSessionProvider');
    return ctx;
};
