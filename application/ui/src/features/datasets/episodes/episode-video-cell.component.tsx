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

    const time = player.time;

    // Make sure webpage renders when video doesn't load correctly
    useEffect(() => {
        const video = videoRef.current;
        const start = episodeVideo.start;

        if (!video) return;
        if (video.readyState < 1) return;
        if (!Number.isFinite(time) || !Number.isFinite(start)) return;

        video.currentTime = time + start;
    }, [time, episodeVideo?.start]);

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
