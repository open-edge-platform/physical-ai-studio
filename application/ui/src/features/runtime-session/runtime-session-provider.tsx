import {
    createContext,
    ReactNode,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
    useSyncExternalStore,
} from 'react';

import { toast } from '@geti-ui/ui';
import useWebSocket from 'react-use-websocket';
import { v4 as uuidv4 } from 'uuid';

import { fetchClient } from '../../api/client';
import {
    SchemaAckMessage,
    SchemaDiscardEpisodeCommand,
    SchemaEnvironmentDraft,
    SchemaEnvironmentWithRelations,
    SchemaErrorMessage,
    SchemaLoadDatasetCommand,
    SchemaLoadEnvironmentCommand,
    SchemaLoadModelCommand,
    SchemaObservationMessage,
    SchemaReloadEnvironmentCommand,
    SchemaSaveEpisodeCommand,
    SchemaSessionState,
    SchemaSetControlCommand,
    SchemaSetTeleopSlowdownCommand,
    SchemaStartEpisodeCommand,
    SchemaStateMessage,
    SchemaStatusMessage,
    SchemaUnloadDatasetCommand,
    SchemaUnloadEnvironmentCommand,
    SchemaUnloadModelCommand,
} from '../../api/openapi-spec';
import { FrameSource } from '../cameras/camera-view';
import { JointsState } from '../robots/use-robot-observations';

/** Session errors go away by themselves; the parts panel keeps showing what is still failed. */
export const ERROR_TOAST = { timeout: 10_000 };

type JsonServerMessage = SchemaAckMessage | SchemaStateMessage | SchemaErrorMessage | SchemaStatusMessage;
/** The last status of each part and device, by component: ``environment``, ``environment/cam 0``, ``control``. */
export type Statuses = Record<string, SchemaStatusMessage>;
type Command =
    | Omit<SchemaLoadEnvironmentCommand, 'request_id'>
    | Omit<SchemaUnloadEnvironmentCommand, 'request_id'>
    | Omit<SchemaReloadEnvironmentCommand, 'request_id'>
    | Omit<SchemaSetControlCommand, 'request_id'>
    | Omit<SchemaSetTeleopSlowdownCommand, 'request_id'>
    | Omit<SchemaLoadDatasetCommand, 'request_id'>
    | Omit<SchemaUnloadDatasetCommand, 'request_id'>
    | Omit<SchemaLoadModelCommand, 'request_id'>
    | Omit<SchemaUnloadModelCommand, 'request_id'>
    | Omit<SchemaStartEpisodeCommand, 'request_id'>
    | Omit<SchemaSaveEpisodeCommand, 'request_id'>
    | Omit<SchemaDiscardEpisodeCommand, 'request_id'>;

/** A model export to load: which model, and the backend and device it runs on. */
export type SessionModelToLoad = Omit<SchemaLoadModelCommand, 'request_id' | 'command'>;

/** The draft a session loads, from a saved environment; its id lets the session load it again (Reload). */
export const environmentDraft = (environment: SchemaEnvironmentWithRelations): SchemaEnvironmentDraft => ({
    id: environment.id,
    name: environment.name,
    robots: (environment.robots ?? []).map(({ robot, tele_operator }) => ({
        robot_id: robot.id,
        tele_operator:
            tele_operator.type === 'robot' ? { type: 'robot', robot_id: tele_operator.robot_id } : { type: 'none' },
    })),
    cameras: (environment.cameras ?? []).flatMap((camera) => (camera.id ? [{ camera_id: camera.id }] : [])),
});

/** Why ``component`` failed, while it is failed. */
export const failureOf = (statuses: Statuses, component: string): string | null => {
    const status = statuses[component];
    return status?.state === 'failed' ? (status.error ?? 'Failed') : null;
};

/** A part that starts or stops takes its devices with it, so their earlier statuses are dropped. */
const withStatus = (statuses: Statuses, status: SchemaStatusMessage): Statuses => {
    const kept =
        status.state === 'starting' || status.state === 'stopped'
            ? Object.fromEntries(
                  Object.entries(statuses).filter(([component]) => !component.startsWith(`${status.component}/`))
              )
            : statuses;
    return { ...kept, [status.component]: status };
};

type Listener<T> = (value: T) => void;

/** Latest value plus listeners, so 20 Hz data reaches its subscribers without React state. */
class Channel<T> {
    value: T | undefined = undefined;
    private listeners = new Set<Listener<T>>();

    publish(value: T) {
        this.value = value;
        this.listeners.forEach((listener) => listener(value));
    }

    subscribe(listener: Listener<T>) {
        this.listeners.add(listener);
        return () => {
            this.listeners.delete(listener);
        };
    }
}

/** The parts a session loads: the environment, and what records from it or runs on it. */
export type SessionPartName = 'environment' | 'dataset' | 'model';

interface RuntimeSessionContextValue {
    state: SchemaSessionState | null;
    /** The last command or worker error, until the next successful load. */
    error: SchemaErrorMessage | null;
    statuses: Statuses;
    /** What the provider is loading for the page: the environment, then the dataset and model it was given. */
    loading: SessionPartName | null;
    isLoading: boolean;
    observations: Channel<SchemaObservationMessage>;
    frame: (key: string) => Channel<Blob>;
    send: (command: Command) => Promise<SchemaAckMessage>;
    /** Load what the page asked for again, after it was refused (a device a closing session still held, say). */
    retry: () => void;
}

const RuntimeSessionContext = createContext<RuntimeSessionContextValue | null>(null);

const SESSION_HZ = 20;

const sessionSocketUrl = (project_id: string) =>
    fetchClient.PATH('/api/projects/{project_id}/runtime/ws', { params: { path: { project_id } } });

/** ``[u32 big-endian header length][header JSON][jpeg bytes, in the order of header.frames]`` */
export const parseObservation = async (
    blob: Blob
): Promise<{ observation: SchemaObservationMessage; frames: Map<string, Blob> }> => {
    const length = new DataView(await blob.slice(0, 4).arrayBuffer()).getUint32(0);
    const observation = JSON.parse(await blob.slice(4, 4 + length).text()) as SchemaObservationMessage;
    const frames = new Map<string, Blob>();
    let offset = 4 + length;
    for (const { key, size } of observation.frames) {
        frames.set(key, blob.slice(offset, offset + size, 'image/jpeg'));
        offset += size;
    }
    return { observation, frames };
};

const commandError = (ack: SchemaAckMessage): SchemaErrorMessage => ({
    type: 'error',
    component: 'session',
    message: ack.error?.message ?? 'The command failed.',
    error_code: ack.error?.error_code ?? 'unknown',
    session_alive: true,
});

/**
 * One runtime session for as long as this provider is mounted.
 *
 * Loads ``environment`` into the session and reloads it whenever it changes. Pages
 * read joints and frames from here and never open their own device sockets.
 */
export const RuntimeSessionProvider = ({
    project_id,
    environment,
    dataset_id,
    model,
    children,
}: {
    project_id: string;
    environment: SchemaEnvironmentDraft | null;
    /** Recorded into once the environment is loaded. */
    dataset_id?: string;
    /** Loaded once the environment is, to run. */
    model?: SessionModelToLoad;
    children: ReactNode;
}) => {
    const [state, setState] = useState<SchemaSessionState | null>(null);
    const [error, setError] = useState<SchemaErrorMessage | null>(null);
    const [statuses, setStatuses] = useState<Statuses>({});
    const [loading, setLoading] = useState<SessionPartName | null>(null);
    const [attempt, setAttempt] = useState(0);
    const retry = useCallback(() => setAttempt((previous) => previous + 1), []);
    const observations = useRef(new Channel<SchemaObservationMessage>()).current;
    const frames = useRef(new Map<string, Channel<Blob>>()).current;
    const pending = useRef(new Map<string, (ack: SchemaAckMessage) => void>()).current;
    const sessionClosed = useRef(false);
    // The latest state, for the load effect to read without rerunning on every state.
    const latestState = useRef<SchemaSessionState | null>(null);
    const frame = useCallback(
        (key: string) => {
            let channel = frames.get(key);
            if (channel === undefined) {
                channel = new Channel<Blob>();
                frames.set(key, channel);
            }
            return channel;
        },
        [frames]
    );

    const { sendJsonMessage } = useWebSocket(sessionSocketUrl(project_id), {
        queryParams: { hz: SESSION_HZ },
        // A new socket is a new session; the load effect below loads it again.
        shouldReconnect: () => !sessionClosed.current,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onClose: () => {
            latestState.current = null;
            setState(null);
            setStatuses({});
            pending.forEach((resolve, request_id) =>
                resolve({
                    type: 'ack',
                    request_id,
                    ok: false,
                    error: { message: 'Connection closed', error_code: 'connection_closed' },
                })
            );
            pending.clear();
        },
        onMessage: (event: MessageEvent) => {
            if (event.data instanceof Blob) {
                void parseObservation(event.data).then(({ observation, frames: jpegs }) => {
                    observations.publish(observation);
                    jpegs.forEach((jpeg, key) => frame(key).publish(jpeg));
                });
                return;
            }
            const message = JSON.parse(event.data) as JsonServerMessage;
            switch (message.type) {
                case 'state':
                    latestState.current = message.data;
                    setState(message.data);
                    break;
                case 'error':
                    sessionClosed.current = !message.session_alive;
                    setError(message);
                    break;
                case 'status':
                    setStatuses((previous) => withStatus(previous, message));
                    if (message.component === 'control' && message.state === 'failed') {
                        // The follower holds; nothing else shows a control.
                        toast.negative(`Control stopped: ${message.error ?? 'it failed'}`, ERROR_TOAST);
                    }
                    break;
                case 'ack':
                    pending.get(message.request_id)?.(message);
                    pending.delete(message.request_id);
                    break;
            }
        },
    });

    const send = useCallback(
        (command: Command) =>
            new Promise<SchemaAckMessage>((resolve) => {
                const request_id = uuidv4();
                pending.set(request_id, resolve);
                sendJsonMessage({ ...command, request_id });
            }),
        [pending, sendJsonMessage]
    );

    // Reload whenever the environment changes, and on every new socket (a new session).
    const environmentKey = JSON.stringify(environment);
    const modelKey = JSON.stringify(model ?? null);
    const hasSession = state !== null;
    useEffect(() => {
        if (!hasSession) {
            return;
        }
        let cancelled = false;
        const draft = JSON.parse(environmentKey) as SchemaEnvironmentDraft | null;
        // A load that replaces the environment keeps a dataset that still fits, and refuses
        // otherwise; loading it again would finish and reopen it.
        const loadDataset = dataset_id !== undefined && latestState.current?.dataset?.dataset_id !== dataset_id;
        // The same goes for a model: loading it again would restart its process.
        const toLoad = JSON.parse(modelKey) as SessionModelToLoad | null;
        const loaded = latestState.current?.model;
        const loadModel =
            toLoad !== null &&
            (loaded?.model_id !== toLoad.model_id ||
                loaded.backend !== toLoad.backend ||
                loaded.device !== toLoad.device);
        const reload = async () => {
            setLoading('environment');
            await send({ command: 'unload_environment' });
            if (draft !== null && draft.robots.length > 0 && !cancelled) {
                let ack = await send({ command: 'load_environment', environment: draft, unload_incompatible: false });
                if (ack.ok && loadDataset && dataset_id !== undefined && !cancelled) {
                    setLoading('dataset');
                    ack = await send({ command: 'load_dataset', dataset_id });
                }
                if (ack.ok && loadModel && toLoad !== null && !cancelled) {
                    setLoading('model');
                    ack = await send({ command: 'load_model', ...toLoad });
                }
                if (!cancelled) {
                    setError(ack.ok ? null : commandError(ack));
                }
            }
            if (!cancelled) {
                setLoading(null);
            }
        };
        void reload();
        return () => {
            cancelled = true;
        };
    }, [environmentKey, dataset_id, modelKey, hasSession, send, attempt]);

    const value = useMemo(
        () => ({ state, error, statuses, loading, isLoading: loading !== null, observations, frame, send, retry }),
        [state, error, statuses, loading, observations, frame, send, retry]
    );

    return <RuntimeSessionContext.Provider value={value}>{children}</RuntimeSessionContext.Provider>;
};

export const useRuntimeSession = () => {
    const context = useContext(RuntimeSessionContext);
    if (context === null) {
        throw new Error('useRuntimeSession must be used within a RuntimeSessionProvider');
    }
    return context;
};

export const useOptionalRuntimeSession = () => useContext(RuntimeSessionContext);

/** The session robot's measured joints, re-rendering the caller at the session rate. */
export const useSessionJoints = (): JointsState => {
    const { observations } = useRuntimeSession();
    const observation = useSyncExternalStore(
        (onChange) => observations.subscribe(onChange),
        () => observations.value
    );
    return useMemo(
        () => Object.entries(observation?.state ?? {}).map(([name, value]) => ({ name, value })),
        [observation]
    );
};

/** Frames of one session camera, by store key. */
export const useSessionCamera = (key: string): FrameSource => {
    const { frame } = useRuntimeSession();
    return frame(key);
};
