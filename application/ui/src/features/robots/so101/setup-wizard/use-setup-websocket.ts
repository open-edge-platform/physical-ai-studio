import { useCallback, useRef, useState } from 'react';

import useWebSocket from 'react-use-websocket';

// ---------------------------------------------------------------------------
// Types — mirrors the backend SO101SetupWorker protocol
// ---------------------------------------------------------------------------

export interface VoltageReading {
    name: string;
    motor_id: number;
    raw: number | null;
}

export interface VoltageResult {
    event: 'voltage_result';
    readings: VoltageReading[];
    avg_voltage: number | null;
    voltage_ok: boolean;
    expected_source: string;
    robot_type: string;
}

export interface MotorProbeEntry {
    name: string;
    motor_id: number;
    found: boolean;
    model_number: number | null;
    model_correct: boolean;
}

export interface CalibrationMotorStatus {
    homing_offset: number;
    range_min: number;
    range_max: number;
    is_calibrated: boolean;
}

export interface MotorProbeResult {
    event: 'motor_probe_result';
    motors: MotorProbeEntry[];
    all_motors_ok: boolean;
    found_count: number;
    total_count: number;
    calibration: {
        motors: Record<string, CalibrationMotorStatus>;
        all_calibrated: boolean;
    } | null;
}

export interface MotorSetupProgress {
    event: 'motor_setup_progress';
    motor: string;
    status: 'scanning' | 'success' | 'error';
    message: string;
}

export interface HomingResult {
    event: 'homing_result';
    offsets: Record<string, number>;
}

export interface MotorPosition {
    position: number;
    min?: number;
    max?: number;
}

export interface PositionsEvent {
    event: 'positions';
    motors: Record<string, MotorPosition>;
}

export interface CalibrationResult {
    event: 'calibration_result';
    calibration: Record<
        string,
        {
            id: number;
            drive_mode: number;
            homing_offset: number;
            range_min: number;
            range_max: number;
        }
    >;
}

export interface StatusEvent {
    event: 'status';
    state: string;
    phase: string;
    message: string;
}

export interface StateWasUpdatedEvent {
    event: 'state_was_updated';
    /** Normalized positions keyed as "{motor_name}.pos" */
    state: Record<string, number>;
}

export interface ErrorEvent {
    event: 'error';
    message: string;
}

export type SetupEvent =
    | VoltageResult
    | MotorProbeResult
    | MotorSetupProgress
    | HomingResult
    | PositionsEvent
    | CalibrationResult
    | StatusEvent
    | StateWasUpdatedEvent
    | ErrorEvent
    | { event: 'pong' };

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

interface UseSetupWebSocketOptions {
    projectId: string;
    robotType: string;
    serialNumber: string;
    enabled?: boolean;
}

export interface SetupWebSocketState {
    /** Current backend phase */
    phase: string | null;
    /** Latest status message */
    statusMessage: string | null;
    /** Voltage check results */
    voltageResult: VoltageResult | null;
    /** Motor probe results */
    probeResult: MotorProbeResult | null;
    /** Motor setup progress per motor */
    motorSetupProgress: Record<string, MotorSetupProgress>;
    /** Homing offsets after centering */
    homingResult: HomingResult | null;
    /** Live positions during range recording */
    positions: PositionsEvent | null;
    /** Final calibration written to motors */
    calibrationResult: CalibrationResult | null;
    /** Normalized joint state for 3D preview (from state_was_updated events) */
    jointState: Record<string, number> | null;
    /** Latest error */
    error: string | null;
    /** Whether the websocket is connected */
    isConnected: boolean;
}

export function useSetupWebSocket({ projectId, robotType, serialNumber, enabled = true }: UseSetupWebSocketOptions) {
    const [state, setState] = useState<SetupWebSocketState>({
        phase: null,
        statusMessage: null,
        voltageResult: null,
        probeResult: null,
        motorSetupProgress: {},
        homingResult: null,
        positions: null,
        calibrationResult: null,
        jointState: null,
        error: null,
        isConnected: false,
    });

    const stateRef = useRef(state);
    stateRef.current = state;

    const handleMessage = useCallback((event: WebSocketEventMap['message']) => {
        try {
            const data = JSON.parse(event.data) as SetupEvent;

            setState((prev) => {
                switch (data.event) {
                    case 'status':
                        return {
                            ...prev,
                            phase: data.phase,
                            statusMessage: data.message,
                            error: null,
                        };

                    case 'voltage_result':
                        return { ...prev, voltageResult: data };

                    case 'motor_probe_result':
                        return { ...prev, probeResult: data };

                    case 'motor_setup_progress':
                        return {
                            ...prev,
                            motorSetupProgress: {
                                ...prev.motorSetupProgress,
                                [data.motor]: data,
                            },
                        };

                    case 'homing_result':
                        return { ...prev, homingResult: data };

                    case 'positions':
                        return { ...prev, positions: data };

                    case 'calibration_result':
                        return { ...prev, calibrationResult: data };

                    case 'state_was_updated':
                        return { ...prev, jointState: data.state };

                    case 'error':
                        return { ...prev, error: data.message };

                    case 'pong':
                        return prev;

                    default:
                        return prev;
                }
            });
        } catch (err) {
            console.error('Failed to parse setup websocket message:', err);
        }
    }, []);

    const url =
        enabled && robotType && serialNumber
            ? `/api/projects/${projectId}/robots/setup/ws` +
              `?robot_type=${encodeURIComponent(robotType)}` +
              `&serial_number=${encodeURIComponent(serialNumber)}`
            : null;

    const { sendJsonMessage, readyState } = useWebSocket(url, {
        onMessage: handleMessage,
        onOpen: () => setState((prev) => ({ ...prev, isConnected: true, error: null })),
        onClose: () => setState((prev) => ({ ...prev, isConnected: false })),
        onError: () => setState((prev) => ({ ...prev, error: 'WebSocket connection error' })),
        shouldReconnect: () => false, // Don't auto-reconnect — user should retry explicitly
    });

    // ------------------------------------------------------------------
    // Command senders
    // ------------------------------------------------------------------

    const startMotorSetup = useCallback(() => {
        sendJsonMessage({ command: 'start_motor_setup' });
    }, [sendJsonMessage]);

    const motorConnected = useCallback(
        (motor: string) => {
            sendJsonMessage({ command: 'motor_connected', motor });
        },
        [sendJsonMessage]
    );

    const finishMotorSetup = useCallback(() => {
        sendJsonMessage({ command: 'finish_motor_setup' });
    }, [sendJsonMessage]);

    const startHoming = useCallback(() => {
        sendJsonMessage({ command: 'start_homing' });
    }, [sendJsonMessage]);

    const startRecording = useCallback(() => {
        sendJsonMessage({ command: 'start_recording' });
    }, [sendJsonMessage]);

    const stopRecording = useCallback(() => {
        sendJsonMessage({ command: 'stop_recording' });
    }, [sendJsonMessage]);

    const reProbe = useCallback(() => {
        // Clear previous results so the UI shows a loading state while rechecking
        setState((prev) => ({ ...prev, voltageResult: null, probeResult: null, error: null }));
        sendJsonMessage({ command: 're_probe' });
    }, [sendJsonMessage]);

    const startPositionsStream = useCallback(() => {
        sendJsonMessage({ command: 'start_positions_stream' });
    }, [sendJsonMessage]);

    const stopPositionsStream = useCallback(() => {
        sendJsonMessage({ command: 'stop_positions_stream' });
    }, [sendJsonMessage]);

    const streamPositions = useCallback(() => {
        sendJsonMessage({ command: 'stream_positions' });
    }, [sendJsonMessage]);

    const stopStream = useCallback(() => {
        sendJsonMessage({ command: 'stop_stream' });
    }, [sendJsonMessage]);

    const ping = useCallback(() => {
        sendJsonMessage({ command: 'ping' });
    }, [sendJsonMessage]);

    return {
        state,
        readyState,
        commands: {
            startMotorSetup,
            motorConnected,
            finishMotorSetup,
            startHoming,
            startRecording,
            stopRecording,
            reProbe,
            startPositionsStream,
            stopPositionsStream,
            streamPositions,
            stopStream,
            ping,
        },
    };
}
