import { useEffect, useState } from 'react';

import { Player } from './use-player';

/** Playhead time that re-renders on every animation frame while the player is playing or seeking. */
export const usePlayerTime = (player: Player) => {
    const [time, setTime] = useState(player.timeRef.current);

    useEffect(() => {
        setTime(player.timeRef.current);
        if (!player.isPlaying && !player.isSeeking) return;

        let raf = 0;
        const tick = () => {
            setTime(player.timeRef.current);
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [player.timeRef, player.isPlaying, player.isSeeking]);

    const seek = (next: number) => {
        player.seek(next);
        setTime(next);
    };

    return { time, seek };
};
