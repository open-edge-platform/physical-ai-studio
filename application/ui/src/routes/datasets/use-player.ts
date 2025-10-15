import { useEffect, useState } from "react";
import { SchemaEpisode } from "../../api/openapi-spec";

export interface Player {
    time: number;
    duration: number;
    isPlaying: boolean;
    play: () => void;
    pause: () => void;
    rewind: () => void;
    seek: (time: number) => void;
}

export const usePlayer = (episode: SchemaEpisode): Player => {
    const [isPlaying, setIsPlaying] = useState(true);
    const [time, setTime] = useState<number>(0);
    const duration = episode.length / episode.fps;
    const frameTime = 1 / episode.fps;

    const play = () => {
        if (time + frameTime > duration) {
            setTime(0);
        }
        setIsPlaying(true);
    }

    const pause = () => {
        setIsPlaying(false);
    }

    const rewind = () => {
        setTime(0);
    }
    const seek = (newTime: number) => {
        setTime(newTime);
    }

    useEffect(() => {
        setTime(0);
        setIsPlaying(true);
    }, [episode])

    useEffect(() => {
        if (isPlaying) {
            const timeAtStart = time;
            const worldTimeAtStart = new Date().getTime() / 1000;
            const interval = setInterval(() => {
                const now = new Date().getTime() / 1000;
                const nextTime = timeAtStart + now - worldTimeAtStart;
                if (nextTime > duration) {
                    setIsPlaying(false);
                }
                setTime(Math.min(nextTime, duration));
            }, frameTime * 1000);
            return () => clearInterval(interval);
        }
    }, [isPlaying]);

    return {
        time,
        duration,
        isPlaying,
        play,
        pause,
        rewind,
        seek,
    }
}

