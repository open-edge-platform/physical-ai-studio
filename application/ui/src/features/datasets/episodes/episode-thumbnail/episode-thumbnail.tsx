import { RefObject, SyntheticEvent, useEffect, useRef, useState } from 'react';

import { Loading } from '@geti-ui/ui';

import classes from './episode-thumbnail.module.css';

export const HOVER_PREVIEW_DELAY_MS = 500;

/**
 * A segment of a (possibly shared) video file. LeRobot v3 stores many episodes back-to-back in
 * one mp4, so `start` / `end` are the episode's boundaries in seconds within that file.
 */
export interface EpisodePreviewVideo {
    url: string;
    start: number;
    end: number;
    fps: number;
}

interface EpisodeThumbnailProps {
    src: string | undefined;
    alt: string;
    isLoading: boolean;
    video: EpisodePreviewVideo | undefined;
}

const restartSegment = (element: HTMLVideoElement, start: number) => {
    element.currentTime = start;
    element.play().catch(() => undefined);
};

const useLoopVideo = (
    videoRef: RefObject<HTMLVideoElement | null>,
    isActive: boolean,
    start: number | undefined,
    end: number | undefined,
    fps: number | undefined
) => {
    useEffect(() => {
        const element = videoRef.current;

        if (!isActive || element === null || start === undefined || end === undefined || fps === undefined) {
            return;
        }

        if (typeof element.requestVideoFrameCallback !== 'function') {
            return;
        }

        const frameDuration = 1 / fps;
        const lastFrameThreshold = end - 1.5 * frameDuration;

        const onFrame: VideoFrameRequestCallback = (_now, { mediaTime }) => {
            if (mediaTime >= lastFrameThreshold) {
                restartSegment(element, start);
            }

            handle = element.requestVideoFrameCallback(onFrame);
        };

        let handle = element.requestVideoFrameCallback(onFrame);

        return () => element.cancelVideoFrameCallback(handle);
    }, [videoRef, isActive, start, end, fps]);
};

export const EpisodeThumbnail = ({ src, alt, isLoading, video }: EpisodeThumbnailProps) => {
    const [failed, setFailed] = useState(false);
    const [isPreviewing, setIsPreviewing] = useState(false);
    const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    const clearHoverTimeout = () => {
        if (hoverTimeoutRef.current !== null) {
            clearTimeout(hoverTimeoutRef.current);
            hoverTimeoutRef.current = null;
        }
    };

    useEffect(() => clearHoverTimeout, []);

    useLoopVideo(videoRef, isPreviewing, video?.start, video?.end, video?.fps);

    if (src === undefined && isLoading) {
        return (
            <div className={classes.episodeImagePlaceholder}>
                <Loading isIndeterminate size='S' aria-label='Loading thumbnail' />
            </div>
        );
    }

    if (src === undefined || failed) {
        return (
            <div role='img' aria-label={alt} className={classes.episodeImagePlaceholder}>
                No preview
            </div>
        );
    }

    const handlePointerEnter = () => {
        if (video === undefined) {
            return;
        }

        clearHoverTimeout();
        hoverTimeoutRef.current = setTimeout(() => setIsPreviewing(true), HOVER_PREVIEW_DELAY_MS);
    };

    const handlePointerLeave = () => {
        clearHoverTimeout();
        setIsPreviewing(false);
    };

    const handleLoadedMetadata = (event: SyntheticEvent<HTMLVideoElement>) => {
        if (video !== undefined) {
            event.currentTarget.currentTime = video.start;
        }
    };

    const handleEnded = (event: SyntheticEvent<HTMLVideoElement>) => {
        if (video !== undefined) {
            restartSegment(event.currentTarget, video.start);
        }
    };

    return (
        <div onPointerEnter={handlePointerEnter} onPointerLeave={handlePointerLeave}>
            {isPreviewing && video !== undefined ? (
                <video
                    ref={videoRef}
                    aria-label={`Video preview: ${alt}`}
                    src={video.url}
                    className={classes.episodeImage}
                    preload='metadata'
                    autoPlay
                    muted
                    playsInline
                    onLoadedMetadata={handleLoadedMetadata}
                    onEnded={handleEnded}
                    onError={() => setIsPreviewing(false)}
                />
            ) : (
                <img alt={alt} src={src} className={classes.episodeImage} onError={() => setFailed(true)} />
            )}
        </div>
    );
};
