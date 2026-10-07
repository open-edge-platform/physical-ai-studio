import { useCallback, useState } from 'react';

import useWebSocket from 'react-use-websocket';

import { SchemaRobotInput } from '../../robot-types';

// ---------------------------------------------------------------------------
// Types — mirrors the backend RobotCalibrationWorker protocol
// ---------------------------------------------------------------------------

export type ZeroCalibrationPhase = 'waiting' | 'connecting' | 'positioning' | 'verification';

interface StatusEvent {
    event: 'status';
    state: string;
    phase: ZeroCalibrationPhase;
    message: string;
}

interface ObservationEvent {
    event: 'observation';
    /** Joint positions in degrees keyed as "{joint_name}.pos" */
    data: Record<string, number>;
}

export interface ZeroCalibrationResult {
    event: 'calibration_result';
    success: boolean;
    /** Joint readings in degrees right after set-zero, keyed as "{joint_name}.pos" */
    joints: Record<string, number>;
    tolerance_deg: number;
}

interface ErrorEvent {
    event: 'error';
    message: string;
    error_code?: string;
}

type CalibrationEvent = StatusEvent | ObservationEvent | ZeroCalibrationResult | ErrorEvent | { event: 'pong' };

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

interface UseZeroCalibrationWebSocketOptions {
    projectId: string;
    /** The unsaved robot to calibrate; the socket opens once it is set. */
    robot: SchemaRobotInput | null;
}

export interface ZeroCalibrationWebSocketState {
    /** Current backend phase */
    phase: ZeroCalibrationPhase | null;
    /** Latest status message; in the positioning phase, the plugin's zero-pose instructions */
    statusMessage: string | null;
    /** Live joint positions for the 3D preview */
    joints: Record<string, number> | null;
    /** Result of the latest set-zero */
    calibrationResult: ZeroCalibrationResult | null;
    /** Whether a set-zero is waiting for its result */
    isSettingZero: boolean;
    /** Latest error */
    error: string | null;
    /** Error code from the backend for contextual error UI */
    errorCode: string | null;
    /** Whether the websocket is connected */
    isConnected: boolean;
}

const INITIAL_STATE: ZeroCalibrationWebSocketState = {
    phase: null,
    statusMessage: null,
    joints: null,
    calibrationResult: null,
    isSettingZero: false,
    error: null,
    errorCode: null,
    isConnected: false,
};

export function useZeroCalibrationWebSocket({ projectId, robot }: UseZeroCalibrationWebSocketOptions) {
    const [state, setState] = useState<ZeroCalibrationWebSocketState>(INITIAL_STATE);

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const data = JSON.parse(event.data) as CalibrationEvent;
            setState((prev) => {
                switch (data.event) {
                    case 'status':
                        return { ...prev, phase: data.phase, statusMessage: data.message };
                    case 'observation':
                        return { ...prev, joints: data.data };
                    case 'calibration_result':
                        return { ...prev, calibrationResult: data, isSettingZero: false };
                    case 'error':
                        return {
                            ...prev,
                            isSettingZero: false,
                            error: data.message,
                            errorCode: data.error_code ?? null,
                        };
                    default:
                        return prev;
                }
            });
        } catch (err) {
            console.error('Failed to parse calibration websocket message:', err);
        }
    }, []);

    const url = robot !== null ? `/api/projects/${projectId}/robots/zero-calibration/ws` : null;

    const { sendJsonMessage, readyState } = useWebSocket(url, {
        onMessage: handleMessage,
        onOpen: () => {
            setState((prev) => ({ ...prev, isConnected: true, error: null, errorCode: null }));
            sendJsonMessage({ command: 'start', robot });
        },
        onClose: (event: WebSocketEventMap['close']) =>
            setState((prev) => ({
                ...prev,
                isConnected: false,
                // Preserve errors already set by an 'error' event; otherwise use the
                // close code to provide a fallback message.
                error:
                    prev.error ?? (event.code !== 1000 ? `Connection closed unexpectedly (code ${event.code})` : null),
                errorCode: prev.errorCode ?? (event.code !== 1000 ? 'connection_closed' : null),
            })),
        onError: () =>
            setState((prev) => ({ ...prev, error: 'WebSocket connection error', errorCode: 'connection_failed' })),
        shouldReconnect: () => false, // Don't auto-reconnect — user should retry explicitly
    });

    // ------------------------------------------------------------------
    // Command senders
    // ------------------------------------------------------------------

    const setZero = useCallback(() => {
        setState((prev) => ({ ...prev, isSettingZero: true, calibrationResult: null, error: null, errorCode: null }));
        sendJsonMessage({ command: 'set_zero' });
    }, [sendJsonMessage]);

    return {
        state,
        readyState,
        commands: { setZero },
    };
}
