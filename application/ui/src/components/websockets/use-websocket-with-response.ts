import { useRef } from 'react';

import useWebSocket, { Options } from 'react-use-websocket';
import { v4 as uuidv4 } from 'uuid';

const DEFAULT_TIMEOUT_MS = 30_000;

interface AckMessage {
    event?: string;
    data?: {
        request_id?: string;
        ok?: boolean;
        error?: string | null;
    };
}

interface ErrorMessage {
    event?: string;
    message?: string;
}

const isAckFor = (message: unknown, requestId: string): message is AckMessage => {
    if (typeof message !== 'object' || message === null) {
        return false;
    }
    const payload = message as AckMessage;
    return payload.event === 'ack' && payload.data?.request_id === requestId;
};

const isErrorEvent = (message: unknown): message is ErrorMessage => {
    if (typeof message !== 'object' || message === null) {
        return false;
    }
    return (message as ErrorMessage).event === 'error';
};

export default function useWebSocketWithResponse(
    url: string | (() => string | Promise<string>) | null,
    options?: Options,
    connect?: boolean
) {
    const messagePromises = useRef<Map<string, (message: MessageEvent) => void>>(new Map());
    const socket = useWebSocket(
        url,
        {
            ...options,
            onMessage: (event) => {
                for (const [_, callback] of messagePromises.current) {
                    callback(event);
                }
                if (options?.onMessage) {
                    options.onMessage(event);
                }
            },
        },
        connect
    );

    const sendJsonMessageAndWait = <MessageType>(
        data: object,
        matcher?: (message: MessageType) => boolean,
        messageOptions?: { timeout?: number }
    ): Promise<MessageType> => {
        const requestId = uuidv4();
        const timeout = messageOptions?.timeout ?? DEFAULT_TIMEOUT_MS;
        socket.sendJsonMessage({ ...data, request_id: requestId });

        return new Promise((resolve, reject) => {
            messagePromises.current.set(requestId, (message) => {
                const messageData = JSON.parse(message.data) as MessageType;
                if (isAckFor(messageData, requestId)) {
                    messagePromises.current.delete(requestId);
                    if (messageData.data?.ok) {
                        resolve(messageData);
                    } else {
                        reject(new Error(messageData.data?.error || 'Runtime request failed.'));
                    }
                    return;
                }
                if (matcher !== undefined && isErrorEvent(messageData)) {
                    messagePromises.current.delete(requestId);
                    reject(new Error(messageData.message || 'Runtime request failed.'));
                    return;
                }
                if (matcher?.(messageData)) {
                    messagePromises.current.delete(requestId);
                    resolve(messageData);
                }
            });
            setTimeout(() => {
                if (messagePromises.current.has(requestId)) {
                    messagePromises.current.delete(requestId);
                    reject(new Error('WebSocket request timed out.'));
                }
            }, timeout);
        });
    };

    return {
        ...socket,
        sendJsonMessageAndWait,
    };
}
