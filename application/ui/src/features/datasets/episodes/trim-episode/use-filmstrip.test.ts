import { describe, expect, it } from '@rstest/core';

import { getMockedEpisodeDetail } from '../../../../test-utils/mocks/mock-episode';
import { DEFAULT_FILMSTRIP_CAMERA, selectCameraVideo } from './use-filmstrip';

const leftVideo = { start: 0, end: 3, path: 'left.mp4' };
const overviewVideo = { start: 1, end: 4, path: 'overview.mp4' };
const rightVideo = { start: 2, end: 5, path: 'right.mp4' };

const { videos } = getMockedEpisodeDetail({
    videos: {
        'observation.images.left': leftVideo,
        'observation.images.overview': overviewVideo,
        'observation.images.right': rightVideo,
    },
});

describe('selectCameraVideo', () => {
    it('defaults to the overview camera', () => {
        expect(DEFAULT_FILMSTRIP_CAMERA).toBe('overview');
        expect(selectCameraVideo(videos, DEFAULT_FILMSTRIP_CAMERA)).toEqual(overviewVideo);
    });

    it('picks the video of the requested camera', () => {
        expect(selectCameraVideo(videos, 'right')).toEqual(rightVideo);
    });

    it('matches the camera name exactly, not as a substring', () => {
        const { videos: wideOnly } = getMockedEpisodeDetail({
            videos: {
                'observation.images.left': leftVideo,
                'observation.images.overview_wide': overviewVideo,
            },
        });

        expect(selectCameraVideo(wideOnly, 'overview')).toEqual(leftVideo);
    });

    it('falls back to the first video when the requested camera is absent', () => {
        expect(selectCameraVideo(videos, 'wrist')).toEqual(leftVideo);
    });

    it('returns undefined when the episode has no videos', () => {
        expect(selectCameraVideo(getMockedEpisodeDetail().videos, 'overview')).toBeUndefined();
    });
});
