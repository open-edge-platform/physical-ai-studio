import { createContext, ReactNode, RefObject, useContext, useRef, useState } from 'react';

import { useMutation, UseMutationResult, useQueryClient } from '@tanstack/react-query';

import { fetchClient } from '../../api/client';
import {
    SchemaDatasetOutput,
    SchemaEnvironmentWithRelations,
    SchemaInferenceDeviceInfo,
    SchemaModel,
} from '../../api/openapi-spec';
import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';

type FollowerSource = 'teleoperation' | 'model' | null;

type InferenceDevice = Pick<SchemaInferenceDeviceInfo, 'backend' | 'device'>;

interface RobotControlState {
    model_loaded: boolean;
    environment_loaded: boolean;
    task_index: number;
    error: boolean;
    is_recording: boolean;
    dataset_loaded: boolean;
    follower_source: FollowerSource;
    episodes_recorded: number;
}

const createRobotControlState = (): RobotControlState => {
    return {
        model_loaded: false,
        environment_loaded: false,
        dataset_loaded: false,
        task_index: 0,
        error: false,
        is_recording: false,
        follower_source: null,
        episodes_recorded: 0,
    };
};

interface RobotControlApiJsonResponse<T> {
    event: string;
    data: T;
}
export interface Observation {
    timestamp: number;
    state: { [joint: string]: number }; // robot joint state before inference
    actions: { [joint: string]: number } | null; // joint actions suggested by inference
    cameras: { [key: string]: string };
}

interface useRobotControlProps {
    children: ReactNode;
    environment: SchemaEnvironmentWithRelations;
    model?: SchemaModel;
    dataset?: SchemaDatasetOutput;
    inferenceDevice?: InferenceDevice;
    onError: (error: string) => void;
}

interface LoadModelPayload {
    model: SchemaModel;
    inference_device: InferenceDevice;
}

type MutationResult<TVariables = void> = UseMutationResult<
    RobotControlApiJsonResponse<RobotControlState>,
    Error,
    TVariables
>;

type RobotControlContextValue = null | {
    observation: RefObject<Observation | undefined>;
    environment: SchemaEnvironmentWithRelations;
    model: SchemaModel | undefined;
    inferenceDevice: InferenceDevice | undefined;
    dataset: SchemaDatasetOutput | undefined;
    state: RobotControlState;
    loadEnvironment: MutationResult<SchemaEnvironmentWithRelations>;
    loadModel: MutationResult<LoadModelPayload>;
    loadDataset: MutationResult<SchemaDatasetOutput>;
    startTask: MutationResult<string>;
    stopTask: MutationResult;
    readyForInference: boolean;
    readyForRecording: boolean;
    setFollowerSource: MutationResult<FollowerSource>;
    startEpisode: MutationResult<string>;
    saveEpisode: MutationResult;
    discardEpisode: MutationResult;
    isConnected: boolean;
};

const RobotControlContext = createContext<RobotControlContextValue>(null);

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

export const RobotControlProvider = (props: useRobotControlProps) => {
    const [state, setState] = useState<RobotControlState>(createRobotControlState());
    const observation = useRef<Observation | undefined>(undefined);

    const [model, setModel] = useState<SchemaModel | undefined>(props.model);
    const [inferenceDevice, setInferenceDevice] = useState<InferenceDevice | undefined>(props.inferenceDevice);
    const [dataset, setDataset] = useState<SchemaDatasetOutput | undefined>(props.dataset);
    const [environment, setEnvironment] = useState<SchemaEnvironmentWithRelations>(props.environment);

    const onOpen = () => {
        loadEnvironment.mutate(props.environment);
        if (model && inferenceDevice) {
            loadModel.mutate({ model, inference_device: inferenceDevice });
        }
        if (dataset) {
            loadDataset.mutate(dataset);
            setFollowerSource.mutate('teleoperation');
        }
    };

    const invalidateEpisodesQuery = useRefreshEpisodes(dataset?.id);
    const { sendJsonMessageAndWait, readyState } = useWebSocketWithResponse(
        fetchClient.PATH('/api/record/robot_control/ws'),
        {
            shouldReconnect: () => true,
            onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
            onError: console.error,
            onClose: () => {
                invalidateEpisodesQuery();

                setState(createRobotControlState());
            },
            onOpen,
        }
    );

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as RobotControlApiJsonResponse<unknown>;
        if (message['event'] === 'observations') {
            observation.current = message['data'] as Observation;
        }
        if (message['event'] === 'state') {
            setState(message['data'] as RobotControlState);
        }

        if (message['event'] === 'error') {
            props.onError(message['data'] as string);
        }
    };

    const loadDataset = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (datasetConfig: SchemaDatasetOutput) => {
            const result = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'load_dataset', data: { dataset: datasetConfig } },
                (data) => data['data']['dataset_loaded']
            );
            setDataset(datasetConfig);
            return result;
        },
    });

    const loadEnvironment = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (env: SchemaEnvironmentWithRelations) => {
            const result = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'load_environment', data: { environment: env } },
                (data) => data['data']['environment_loaded']
            );
            setEnvironment(env);
            return result;
        },
    });

    const loadModel = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (properties: LoadModelPayload) => {
            const result = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'load_model', data: properties },
                ({ data }) => data['model_loaded']
            );
            setModel(properties.model);
            setInferenceDevice(properties.inference_device);
            return result;
        },
    });

    const startTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (task: string) =>
            await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'start_task', data: { task } },
                ({ data }) => data['follower_source'] === 'model'
            ),
    });

    const stopTask = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () =>
            await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'stop_task', data: {} },
                ({ data }) => data['follower_source'] === null
            ),
    });

    const startEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (task: string) => {
            const message = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'start_recording', data: { task } },
                ({ data }) => data['is_recording'] == true
            );
            return message;
        },
    });

    const saveEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () => {
            const message = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'save_episode', data: {} },
                ({ data }) => data['is_recording'] == false
            );
            return message;
        },
    });

    const discardEpisode = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async () => {
            const message = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'discard_episode', data: {} },
                ({ data }) => data['is_recording'] == false
            );
            return message;
        },
    });

    const setFollowerSource = useMutation({
        meta: { skipInvalidation: true },
        mutationFn: async (follower_source: FollowerSource) => {
            const message = await sendJsonMessageAndWait<RobotControlApiJsonResponse<RobotControlState>>(
                { event: 'set_follower_source', data: { follower_source } },
                ({ data }) => data['follower_source'] == follower_source
            );
            return message;
        },
    });

    const readyForInference = state.environment_loaded && state.model_loaded;
    const readyForRecording = state.environment_loaded && state.dataset_loaded;
    const isConnected = readyState === 1;

    return (
        <RobotControlContext.Provider
            value={{
                observation,
                environment,
                dataset,
                model,
                inferenceDevice,
                state,
                loadEnvironment,
                loadModel,
                loadDataset,
                startTask,
                stopTask,
                readyForInference,
                readyForRecording,
                setFollowerSource,
                startEpisode,
                saveEpisode,
                discardEpisode,
                isConnected,
            }}
        >
            {props.children}
        </RobotControlContext.Provider>
    );
};

export const useRobotControl = () => {
    const ctx = useContext(RobotControlContext);
    if (!ctx) throw new Error('useRobotControl must be used within RobotControlProvider');
    return ctx;
};
