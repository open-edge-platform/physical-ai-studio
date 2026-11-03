import { rs } from '@rstest/core';
import { act, fireEvent, screen } from '@testing-library/react';

import { render } from '../../../../test-utils/render';
import { EpisodeThumbnail, HOVER_PREVIEW_DELAY_MS } from './episode-thumbnail';

const ALT = 'Camera frame of 1';
const THUMBNAIL_SRC = 'http://localhost/thumbnail.png';
const VIDEO_URL = 'http://localhost/video.mp4';
const VIDEO = { url: VIDEO_URL, start: 13, end: 23, fps: 10 };
const LAST_FRAME_TIME = 22.9;

const renderThumbnail = (props: Partial<Parameters<typeof EpisodeThumbnail>[0]> = {}) =>
    render(<EpisodeThumbnail alt={ALT} src={THUMBNAIL_SRC} isLoading={false} video={VIDEO} {...props} />);

const queryVideo = () => screen.queryByLabelText(`Video preview: ${ALT}`);

const hoverThumbnail = () => fireEvent.pointerEnter(screen.getByRole('img', { name: ALT }));

const startPreview = () => {
    hoverThumbnail();
    act(() => rs.advanceTimersByTime(HOVER_PREVIEW_DELAY_MS));

    return queryVideo() as HTMLVideoElement;
};

const pendingFrameCallbacks = new Map<number, VideoFrameRequestCallback>();

const installFakeVideoFrameCallback = () => {
    let nextFrameHandle = 1;

    HTMLVideoElement.prototype.requestVideoFrameCallback = rs.fn((callback: VideoFrameRequestCallback) => {
        const handle = nextFrameHandle++;
        pendingFrameCallbacks.set(handle, callback);
        return handle;
    });
    HTMLVideoElement.prototype.cancelVideoFrameCallback = rs.fn((handle: number) => {
        pendingFrameCallbacks.delete(handle);
    });
};

const uninstallFakeVideoFrameCallback = () => {
    const prototype = HTMLVideoElement.prototype as Partial<HTMLVideoElement>;
    delete prototype.requestVideoFrameCallback;
    delete prototype.cancelVideoFrameCallback;
    pendingFrameCallbacks.clear();
};

const presentFrame = (video: HTMLVideoElement, mediaTime: number) => {
    video.currentTime = mediaTime;
    const callbacks = [...pendingFrameCallbacks.values()];
    pendingFrameCallbacks.clear();

    act(() =>
        callbacks.forEach((callback) =>
            callback(performance.now(), {
                mediaTime,
                presentationTime: 0,
                expectedDisplayTime: 0,
                width: 320,
                height: 240,
                presentedFrames: 0,
            })
        )
    );
};

describe('EpisodeThumbnail', () => {
    beforeEach(() => {
        rs.useFakeTimers();
        installFakeVideoFrameCallback();
        rs.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    });

    afterEach(() => {
        rs.useRealTimers();
        rs.restoreAllMocks();
        pendingFrameCallbacks.clear();
    });

    afterAll(uninstallFakeVideoFrameCallback);

    it('shows a loading indicator while the thumbnail camera is loading', () => {
        renderThumbnail({ src: undefined, isLoading: true });

        expect(screen.getByLabelText('Loading thumbnail')).toBeInTheDocument();
    });

    it('shows a "No preview" placeholder when there is no thumbnail source', () => {
        renderThumbnail({ src: undefined });

        expect(screen.getByRole('img', { name: ALT })).toHaveTextContent('No preview');
    });

    it('shows a "No preview" placeholder when the thumbnail image fails to load', () => {
        renderThumbnail();

        fireEvent.error(screen.getByRole('img', { name: ALT }));

        expect(screen.getByRole('img', { name: ALT })).toHaveTextContent('No preview');
    });

    it('plays the video preview after hovering for the preview delay', () => {
        renderThumbnail();

        expect(screen.getByRole('img', { name: ALT })).toHaveAttribute('src', THUMBNAIL_SRC);

        hoverThumbnail();
        act(() => rs.advanceTimersByTime(HOVER_PREVIEW_DELAY_MS - 1));

        expect(queryVideo()).not.toBeInTheDocument();

        act(() => rs.advanceTimersByTime(1));

        expect(queryVideo()).toHaveAttribute('src', VIDEO_URL);
        expect(screen.queryByRole('img', { name: ALT })).not.toBeInTheDocument();
    });

    it('does not show the video preview when the pointer leaves before the delay', () => {
        renderThumbnail();

        hoverThumbnail();
        act(() => rs.advanceTimersByTime(HOVER_PREVIEW_DELAY_MS / 2));
        fireEvent.pointerLeave(screen.getByRole('img', { name: ALT }));
        act(() => rs.advanceTimersByTime(HOVER_PREVIEW_DELAY_MS));

        expect(queryVideo()).not.toBeInTheDocument();
        expect(screen.getByRole('img', { name: ALT })).toBeInTheDocument();
    });

    it('returns to the thumbnail image and stops watching frames when the pointer leaves the preview', () => {
        renderThumbnail();
        const video = startPreview();
        expect(pendingFrameCallbacks.size).toBe(1);

        fireEvent.pointerLeave(video);

        expect(queryVideo()).not.toBeInTheDocument();
        expect(screen.getByRole('img', { name: ALT })).toBeInTheDocument();
        expect(HTMLVideoElement.prototype.cancelVideoFrameCallback).toHaveBeenCalled();
        expect(pendingFrameCallbacks.size).toBe(0);
    });

    it('keeps the thumbnail image when there is no video for the episode', () => {
        renderThumbnail({ video: undefined });

        hoverThumbnail();
        act(() => rs.advanceTimersByTime(HOVER_PREVIEW_DELAY_MS));

        expect(queryVideo()).not.toBeInTheDocument();
        expect(screen.getByRole('img', { name: ALT })).toBeInTheDocument();
    });

    it('seeks to the episode start once the video metadata is loaded', () => {
        renderThumbnail();
        const video = startPreview();

        fireEvent.loadedMetadata(video);

        expect(video.currentTime).toBe(VIDEO.start);
    });

    it('loops the episode by restarting on its last frame, before the next episode is shown', () => {
        renderThumbnail();
        const video = startPreview();

        presentFrame(video, LAST_FRAME_TIME - 1 / VIDEO.fps);

        expect(HTMLMediaElement.prototype.play).not.toHaveBeenCalled();

        presentFrame(video, LAST_FRAME_TIME);

        expect(video.currentTime).toBe(VIDEO.start);
        expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1);

        presentFrame(video, VIDEO.start + 1);
        presentFrame(video, LAST_FRAME_TIME);

        expect(video.currentTime).toBe(VIDEO.start);
        expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(2);
    });

    it('restarts the episode when the file ends if requestVideoFrameCallback is unsupported', () => {
        uninstallFakeVideoFrameCallback();
        renderThumbnail();
        const video = startPreview();

        video.currentTime = VIDEO.end - 0.01;
        fireEvent.ended(video);

        expect(video.currentTime).toBe(VIDEO.start);
        expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
    });
});
