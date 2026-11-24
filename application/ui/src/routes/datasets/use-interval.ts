import { useEffect, useRef } from 'react';

/** Calls `callback` every `delay` ms; `null` pauses it. */
export const useInterval = (callback: () => void, delay: number | null) => {
    const savedCallback = useRef<() => void>(callback);

    useEffect(() => {
        savedCallback.current = callback;
    }, [callback]);

    useEffect(() => {
        function tick() {
            savedCallback.current();
        }
        if (delay !== null) {
            const id = setInterval(tick, delay);
            return () => clearInterval(id);
        }
    }, [delay]);
};
