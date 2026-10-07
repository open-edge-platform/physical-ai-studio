import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SchemaRobotInput } from '../../robot-types';
import { useZeroCalibrationWebSocket } from './use-zero-calibration-websocket';

const sendJsonMessage = vi.fn();
let capturedUrl: unknown;
let capturedOptions: { onMessage?: (event: MessageEvent) => void; onOpen?: () => void } = {};

vi.mock('react-use-websocket', () => ({
    default: (url: unknown, options: typeof capturedOptions) => {
        capturedUrl = url;
        capturedOptions = options;
        return { sendJsonMessage, readyState: 1 };
    },
}));

const deliver = (payload: unknown) => {
    capturedOptions.onMessage?.({ data: JSON.stringify(payload) } as MessageEvent);
};

const robot = {
    id: 'robot-1',
    name: 'Khaos',
    type: 'ReBot_B601_RS_Follower',
    payload: { connection_string: 'can0' },
} as unknown as SchemaRobotInput;

describe('useZeroCalibrationWebSocket', () => {
    afterEach(() => {
        sendJsonMessage.mockClear();
    });

    it('does not open a socket until there is a robot to calibrate', () => {
        renderHook(() => useZeroCalibrationWebSocket({ projectId: 'project-1', robot: null }));

        expect(capturedUrl).toBeNull();
    });

    it('starts calibration with the unsaved robot once the socket opens', () => {
        renderHook(() => useZeroCalibrationWebSocket({ projectId: 'project-1', robot }));

        act(() => capturedOptions.onOpen?.());

        expect(capturedUrl).toBe('/api/projects/project-1/robots/zero-calibration/ws');
        expect(sendJsonMessage).toHaveBeenCalledWith({ command: 'start', robot });
    });

    it('tracks instructions and live joints from the worker', () => {
        const { result } = renderHook(() => useZeroCalibrationWebSocket({ projectId: 'project-1', robot }));

        act(() => {
            deliver({ event: 'status', state: 'running', phase: 'positioning', message: 'Fold the arm.' });
            deliver({ event: 'observation', data: { 'shoulder_pan.pos': 12.5 } });
        });

        expect(result.current.state.phase).toBe('positioning');
        expect(result.current.state.statusMessage).toBe('Fold the arm.');
        expect(result.current.state.joints).toEqual({ 'shoulder_pan.pos': 12.5 });
    });

    it('sends set_zero and stores the result', () => {
        const { result } = renderHook(() => useZeroCalibrationWebSocket({ projectId: 'project-1', robot }));

        act(() => result.current.commands.setZero());

        expect(sendJsonMessage).toHaveBeenCalledWith({ command: 'set_zero' });
        expect(result.current.state.isSettingZero).toBe(true);

        const calibrationResult = {
            event: 'calibration_result',
            success: true,
            joints: { 'shoulder_pan.pos': 0.2 },
            tolerance_deg: 5,
        };
        act(() => deliver(calibrationResult));

        expect(result.current.state.isSettingZero).toBe(false);
        expect(result.current.state.calibrationResult).toEqual(calibrationResult);
    });

    it('keeps the error code from an error event', () => {
        const { result } = renderHook(() => useZeroCalibrationWebSocket({ projectId: 'project-1', robot }));

        act(() => deliver({ event: 'error', message: 'No such device', error_code: 'device_not_found' }));

        expect(result.current.state.error).toBe('No such device');
        expect(result.current.state.errorCode).toBe('device_not_found');
    });
});
