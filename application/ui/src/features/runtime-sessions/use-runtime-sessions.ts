import { $api } from '../../api/client';
import { SchemaRuntimeSessionInfo } from '../../api/openapi-spec';

/**
 * Runtime sessions are detached processes the backend cannot push events about,
 * so both of these poll. The two intervals differ on purpose: the count is a
 * directory read and is mounted on every page, the list opens a transport
 * session per runtime session and is only mounted while someone is looking.
 */
const COUNT_POLL_MS = 5_000;
const LIST_POLL_MS = 2_000;

export const useRuntimeSessionCount = () =>
    $api.useQuery('get', '/api/runtime/sessions/count', {}, { refetchInterval: COUNT_POLL_MS });

export const useRuntimeSessions = () =>
    $api.useQuery('get', '/api/runtime/sessions', {}, { refetchInterval: LIST_POLL_MS });

export const useStopRuntimeSession = () =>
    $api.useMutation('post', '/api/runtime/sessions/{session_name}/stop', {
        meta: {
            invalidates: [
                ['get', '/api/runtime/sessions'],
                ['get', '/api/runtime/sessions/count'],
            ],
        },
    });

/** What the session is doing, not merely that it is up. */
export const sessionActivity = (session: SchemaRuntimeSessionInfo): string => {
    if (!session.activity) {
        return session.status;
    }
    return session.activity.is_recording ? 'recording' : session.activity.follower_source;
};

export type SessionStatusVariant = 'positive' | 'negative' | 'notice' | 'neutral';

/**
 * Color means attention, not control mode. Green = doing work, yellow = idle
 * or starting, red = broken, gray = stopped. Hold and teleop stay distinct
 * without a fourth hue for policy vs recording.
 */
export const sessionStatusVariant = (session: SchemaRuntimeSessionInfo): SessionStatusVariant => {
    switch (session.status) {
        case 'error':
        case 'unreachable':
            return 'negative';
        case 'stopped':
            return 'neutral';
        case 'starting':
            return 'notice';
        case 'running':
            return sessionActivity(session) === 'hold' ? 'notice' : 'positive';
        default: {
            // No fallthrough on purpose: a status added to the contract makes
            // this assignment fail to compile, so it has to be classified above
            // rather than silently inheriting the running colour.
            const unclassified: never = session.status;
            return unclassified ?? 'neutral';
        }
    }
};

/** What to call a session in the UI, falling back to its raw name for an orphan. */
export const sessionLabel = (session: SchemaRuntimeSessionInfo): string =>
    session.follower_name ?? session.session_name;

/** The session driving a robot, if one is running. */
export const sessionForRobot = (
    sessions: SchemaRuntimeSessionInfo[] | undefined,
    robotId: string
): SchemaRuntimeSessionInfo | undefined => sessions?.find((session) => session.session_name === `rt-${robotId}`);

/** Seconds until an unattached session shuts itself down, or undefined when someone is watching. */
export const idleSecondsRemaining = (session: SchemaRuntimeSessionInfo, now: number): number | undefined => {
    if (session.attached !== false || !session.idle_deadline) {
        return undefined;
    }
    return Math.max(0, Math.round((new Date(session.idle_deadline).getTime() - now) / 1000));
};

export const uptimeLabel = (session: SchemaRuntimeSessionInfo, now: number): string | undefined => {
    if (!session.started_at) {
        return undefined;
    }
    const seconds = Math.max(0, Math.floor((now - new Date(session.started_at).getTime()) / 1000));
    if (seconds < 60) {
        return `${seconds}s`;
    }
    if (seconds < 3600) {
        return `${Math.floor(seconds / 60)}m`;
    }
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
};
