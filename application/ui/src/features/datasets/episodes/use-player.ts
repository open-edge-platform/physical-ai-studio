import { RefObject, useEffect, useRef, useState } from 'react';

import { SchemaEpisode } from '../../../api/openapi-spec';

export enum PlayState {
    Paused,
    Playing,
    Seeking,
}

export interface Player {
    timeRef: RefObject<number>;
    duration: number;
    state: PlayState;
    isPlaying: boolean;
    isPaused: boolean;
    isSeeking: boolean;
    play: () => void;
    pause: () => void;
    rewind: () => void;
    seek: (time: number) => void;
}

export const usePlayer = (episode: SchemaEpisode): Player => {
    const [state, setState] = useState<PlayState>(PlayState.Paused);
    const timeRef = useRef(0);
    const duration = episode.length / episode.fps;
    const frameTime = 1 / episode.fps;

    const setTimeSynced = (newTime: number) => {
        timeRef.current = newTime;
    };

    const play = () => {
        if (timeRef.current + frameTime > duration) {
            setTimeSynced(0);
        }
        setState(PlayState.Playing);
    };

    const pause = () => {
        setState(PlayState.Paused);
    };

    const rewind = () => {
        setTimeSynced(0);
    };

    const seek = (newTime: number) => {
        setTimeSynced(newTime);
        setState(PlayState.Seeking);
    };

    useEffect(() => {
        setTimeSynced(0);
        setState(PlayState.Paused);
    }, [episode]);

    useEffect(() => {
        if (state === PlayState.Playing) {
            const timeAtStart = timeRef.current;
            const worldTimeAtStart = new Date().getTime() / 1000;
            const interval = setInterval(() => {
                const now = new Date().getTime() / 1000;
                const nextTime = timeAtStart + now - worldTimeAtStart;
                if (nextTime > duration) {
                    pause();
                }

                timeRef.current = Math.min(nextTime, duration);
            }, frameTime * 1000);
            return () => clearInterval(interval);
        }
    }, [state, duration, frameTime]);

    return {
        isPlaying: state === PlayState.Playing,
        isSeeking: state === PlayState.Seeking,
        isPaused: state === PlayState.Paused,
        timeRef,
        duration,
        state,
        play,
        pause,
        rewind,
        seek,
    };
};
