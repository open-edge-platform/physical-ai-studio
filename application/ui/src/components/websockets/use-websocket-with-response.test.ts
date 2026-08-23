import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import useWebSocketWithResponse from './use-websocket-with-response';

const sendJsonMessage = vi.fn();
let capturedOnMessage: ((event: MessageEvent) => void) | undefined;

vi.mock('react-use-websocket', () => ({
    default: (_url: unknown, options: { onMessage?: (event: MessageEvent) => void }) => {
        capturedOnMessage = options.onMessage;
        return { sendJsonMessage, readyState: 1 };
    },
}));

const deliver = (payload: unknown) => {
    capturedOnMessage?.({ data: JSON.stringify(payload) } as MessageEvent);
};

// Placeholder only — react-use-websocket is mocked; no socket is opened.
const SOCKET_URL = 'wss://runtime';

describe('useWebSocketWithResponse', () => {
    afterEach(() => {
        sendJsonMessage.mockClear();
        vi.clearAllTimers();
        vi.useRealTimers();
    });

    it('puts request_id on the sent payload', () => {
        vi.useFakeTimers();
        const { result } = renderHook(() => useWebSocketWithResponse(SOCKET_URL));

        void result.current.sendJsonMessageAndWait({ event: 'load_model' }, () => false);

        expect(sendJsonMessage).toHaveBeenCalledTimes(1);
        const sent = sendJsonMessage.mock.calls[0][0] as { event: string; request_id: string };
        expect(sent.event).toBe('load_model');
        expect(sent.request_id).toEqual(expect.any(String));
    });

    it('resolves when an ack with a matching id arrives', async () => {
        const { result } = renderHook(() => useWebSocketWithResponse(SOCKET_URL));
        const pending = result.current.sendJsonMessageAndWait({ event: 'save_episode' });
        const requestId = (sendJsonMessage.mock.calls[0][0] as { request_id: string }).request_id;

        act(() => {
            deliver({ event: 'ack', data: { request_id: requestId, ok: true } });
        });

        await expect(pending).resolves.toEqual({ event: 'ack', data: { request_id: requestId, ok: true } });
    });

    it('rejects when an ack reports ok: false', async () => {
        const { result } = renderHook(() => useWebSocketWithResponse(SOCKET_URL));
        const pending = result.current.sendJsonMessageAndWait({ event: 'save_episode' });
        const requestId = (sendJsonMessage.mock.calls[0][0] as { request_id: string }).request_id;

        act(() => {
            deliver({ event: 'ack', data: { request_id: requestId, ok: false, error: 'episode folder is locked' } });
        });

        await expect(pending).rejects.toThrow('episode folder is locked');
    });

    it('resolves publication commands on a matcher rather than an ack', async () => {
        const { result } = renderHook(() => useWebSocketWithResponse(SOCKET_URL));
        const pending = result.current.sendJsonMessageAndWait<{ event: string; data?: { model_loaded?: boolean } }>(
            { event: 'load_model' },
            (message) => message.event === 'state' && message.data?.model_loaded === true
        );

        act(() => {
            deliver({ event: 'state', data: { model_loaded: true, follower_source: 'hold' } });
        });

        await expect(pending).resolves.toMatchObject({ event: 'state', data: { model_loaded: true } });
    });

    it('rejects a matcher waiter when an error event arrives', async () => {
        const { result } = renderHook(() => useWebSocketWithResponse(SOCKET_URL));
        const pending = result.current.sendJsonMessageAndWait<{ event: string; data?: { follower_source?: string } }>(
            { event: 'start_task' },
            (message) => message.event === 'state' && message.data?.follower_source === 'policy'
        );

        act(() => {
            deliver({ event: 'error', message: 'No policy is loaded.', error_code: 'policy_not_loaded' });
        });

        await expect(pending).rejects.toThrow('No policy is loaded.');
    });

    it('does not reject an ack-only waiter on an unrelated error event', async () => {
        const { result } = renderHook(() => useWebSocketWithResponse(SOCKET_URL));
        const pending = result.current.sendJsonMessageAndWait({ event: 'save_episode' });
        const requestId = (sendJsonMessage.mock.calls[0][0] as { request_id: string }).request_id;

        act(() => {
            deliver({ event: 'error', message: 'The leader robot is not responding.' });
        });
        act(() => {
            deliver({ event: 'ack', data: { request_id: requestId, ok: true } });
        });

        await expect(pending).resolves.toEqual({ event: 'ack', data: { request_id: requestId, ok: true } });
    });

    it('rejects after the default timeout', async () => {
        vi.useFakeTimers();
        const { result } = renderHook(() => useWebSocketWithResponse(SOCKET_URL));
        const pending = result.current.sendJsonMessageAndWait({ event: 'load_model' }, () => false);
        const assertion = expect(pending).rejects.toThrow('WebSocket request timed out.');

        await act(async () => {
            vi.advanceTimersByTime(30_000);
        });

        await assertion;
    });
});
