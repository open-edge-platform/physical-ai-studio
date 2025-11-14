import { useEffect, useRef } from 'react';

export const useInterval = (callback: () => void, delay: number) => {
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
