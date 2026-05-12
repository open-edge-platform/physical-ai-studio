import { useEffect, useRef } from 'react';

import { fetchClient } from '../../../api/client';
import { SchemaEpisodeVideo } from '../../../api/openapi-spec';
import { useFittedMediaSize } from '../../cameras/use-fitted-media-size';
import { useEpisodeViewer } from './episode-viewer-provider.component';

export const EpisodeVideoCell = ({
    episodeVideo,
    datasetId,
}: {
    episodeVideo: SchemaEpisodeVideo;
    datasetId: string;
}) => {
    const { player } = useEpisodeViewer();
    const url = fetchClient.PATH('/api/dataset/{dataset_id}/video/{video_path}', {
        params: {
            path: {
                dataset_id: datasetId,
                video_path: episodeVideo.path,
            },
        },
    });

    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        const video = videoRef.current;
        const start = episodeVideo.start;
        if (!video || !Number.isFinite(start)) return;

        video.currentTime = player.timeRef.current + start;
        if (player.isPlaying) {
            video.play();
        } else {
            video.pause();
        }
    }, [player, episodeVideo.start, videoRef]);

    useEffect(() => {
        const video = videoRef.current;
        if (video && player.isSeeking) {
            const interval = setInterval(() => {
                video.currentTime = player.timeRef.current;
            }, 1000 / 60);
            return () => clearInterval(interval);
        }
    }, [player, videoRef]);

    const { containerRef, width, height } = useFittedMediaSize(
        videoRef.current?.videoWidth,
        videoRef.current?.videoHeight
    );

    /* eslint-disable jsx-a11y/media-has-caption */
    return (
        <div ref={containerRef} style={{ height: '100%', width: '100%' }}>
            <video ref={videoRef} src={url} width={width} height={height} />
        </div>
    );
};
