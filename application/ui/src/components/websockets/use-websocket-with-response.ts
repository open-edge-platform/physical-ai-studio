import { useRef } from 'react';

import useWebSocket, { Options } from 'react-use-websocket';
import { v4 as uuidv4 } from 'uuid';

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

    const sendJsonMessageAndWait = async (
        data: object,
        matcher: (message: MessageEvent) => boolean,
        messageOptions?: { timeout: number }
    ) => {
        const requestId = uuidv4();
        socket.sendJsonMessage(data);

        return new Promise((resolve, reject) => {
            messagePromises.current.set(requestId, (message) => {
                console.log("parsing", message.data)
                const data = JSON.parse(message.data)
                console.log(data);
                if (matcher(data)) {
                    messagePromises.current.delete(requestId);
                    resolve(data);
                }
            });
            if (messageOptions?.timeout)
                setTimeout(() => {
                    if (messagePromises.current.has(requestId)) {
                        messagePromises.current.delete(requestId);
                        reject(new Error('WebSocket request timed out.'));
                    }
                }, messageOptions?.timeout);
        });
    };

    return {
        ...socket,
        sendJsonMessageAndWait,
    };
}
