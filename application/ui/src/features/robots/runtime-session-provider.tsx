import { createContext, ReactNode, RefObject, useContext, useRef, useState } from 'react';

import { useMutation, UseMutationResult, useQueryClient } from '@tanstack/react-query';

import {
    SchemaDatasetOutput,
    SchemaEnvironmentWithRelations,
    SchemaInferenceDeviceInfo,
    SchemaModel,
} from '../../api/openapi-spec';
import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';
import { useProjectId } from '../projects/use-project';
import { FollowerSource, runtimeSocketUrl } from './use-joint-state';

type InferenceDevice = Pick<SchemaInferenceDeviceInfo, 'backend' | 'device'>;

interface RuntimeSessionState {
    connected: boolean;
    follower_source: FollowerSource;
    model_loaded: boolean;
    task: string | null;
    dataset_loaded: boolean;
    is_recording: boolean;
    episodes_recorded: number;
}

const createRuntimeSessionState = (): RuntimeSessionState => ({
    connected: false,
    follower_source: 'hold',
    model_loaded: false,
    task: null,
    dataset_loaded: false,
    is_recording: false,
    episodes_recorded: 0,
});

interface RuntimeApiJsonResponse<T = RuntimeSessionState> {
    event: string;
    data?: T;
    actions?: Record<string, number> | null;
    message?: string;
    error_code?: string;
}

interface RuntimeSessionProviderProps {
    children: ReactNode;
    environment: SchemaEnvironmentWithRelations;
    model?: SchemaModel;
    dataset?: SchemaDatasetOutput;
    inferenceDevice?: InferenceDevice;
    onError: (error: string) => void;
}

type MutationResult<TVariables = void> = UseMutationResult<RuntimeApiJsonResponse, Error, TVariables>;

type RuntimeSessionContextValue = {
    observation: RefObject<Record<string, number> | undefined>;
    actions: RefObject<Record<string, number> | undefined>;
    environment: SchemaEnvironmentWithRelations;
    model: SchemaModel | undefined;
    dataset: SchemaDatasetOutput | undefined;
    inferenceDevice: InferenceDevice | undefined;
    state: RuntimeSessionState;
    loadModel: MutationResult<{ model: SchemaModel; inference_device: InferenceDevice }>;
    loadDataset: MutationResult<SchemaDatasetOutput>;
    startTask: MutationResult<string>;
    stopTask: MutationResult;
    setFollowerSource: MutationResult<FollowerSource>;
    startEpisode: MutationResult<string>;
    saveEpisode: MutationResult;
    discardEpisode: MutationResult;
    readyForInference: boolean;
    readyForRecording: boolean;
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

const EPISODE_ACK_TIMEOUT_MS = 60_000;

const useRefreshEpisodes = (dataset_id?: string) => {
    const queryClient = useQueryClient();

    return () => {
        if (dataset_id === undefined) {
            return;
        }
        queryClient.invalidateQueries({
            queryKey: [
                'get',
                '/api/dataset/{dataset_id}/episodes',
                {
                    params: { path: { dataset_id } },
                },
            ],
        });
    };
};

export const RuntimeSessionProvider = (props: RuntimeSessionProviderProps) => {
    const { project_id } = useProjectId();
    const [state, setState] = useState<RuntimeSessionState>(createRuntimeSessionState());
    const observation = useRef<Record<string, number> | undefined>(undefined);
    const actions = useRef<Record<string, number> | undefined>(undefined);
    const [model, setModel] = useState<SchemaModel | undefined>(props.model);
    const [inferenceDevice, setInferenceDevice] = useState<InferenceDevice | undefined>(props.inferenceDevice);
    const [dataset, setDataset] = useState<SchemaDatasetOutput | undefined>(props.dataset);
    const devices = handshakeDevices(props.environment);
    const invalidateEpisodesQuery = useRefreshEpisodes(dataset?.id);

    const onOpen = () => {
        socket.sendJsonMessage({
            follower_id: devices.follower_id,
            leader_id: devices.leader_id,
            camera_ids: devices.camera_ids,
        });
        if (props.model && props.inferenceDevice) {
            loadModel.mutate({ model: props.model, inference_device: props.inferenceDevice });
        }
        if (props.dataset) {
            loadDataset.mutate(props.dataset);
            setFollowerSource.mutate('teleop');
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
                actions.current = message.actions ?? undefined;
            }
            if (message.event === 'state' && message.data !== undefined && typeof message.data === 'object') {
                const next = message.data as Partial<RuntimeSessionState>;
                setState({
                    connected: next.connected ?? false,
                    follower_source: next.follower_source ?? 'hold',
                    model_loaded: next.model_loaded ?? false,
                    task: next.task ?? null,
                    dataset_loaded: next.dataset_loaded ?? false,
                    is_recording: next.is_recording ?? false,
                    episodes_recorded: next.episodes_recorded ?? 0,
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

    const loadDataset = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (datasetConfig: SchemaDatasetOutput) => {
            const result = await socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'load_dataset', data: { dataset_id: datasetConfig.id } },
                ({ event, data }) => event === 'state' && data?.dataset_loaded === true
            );
            setDataset(datasetConfig);
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

    const startEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (task: string) =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'start_recording', data: { task } },
                ({ event, data }) => event === 'state' && data?.is_recording === true
            ),
    });

    const saveEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () => {
            const result = await socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>(
                { event: 'save_episode', data: {} },
                undefined,
                { timeout: EPISODE_ACK_TIMEOUT_MS }
            );
            invalidateEpisodesQuery();
            return result;
        },
    });

    const discardEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () =>
            socket.sendJsonMessageAndWait<RuntimeApiJsonResponse>({ event: 'discard_episode', data: {} }, undefined, {
                timeout: EPISODE_ACK_TIMEOUT_MS,
            }),
    });

    return (
        <RuntimeSessionContext.Provider
            value={{
                observation,
                actions,
                environment: props.environment,
                model,
                dataset,
                inferenceDevice,
                state,
                loadModel,
                loadDataset,
                startTask,
                stopTask,
                setFollowerSource,
                startEpisode,
                saveEpisode,
                discardEpisode,
                readyForInference: state.connected && state.model_loaded,
                readyForRecording: state.connected && state.dataset_loaded,
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
