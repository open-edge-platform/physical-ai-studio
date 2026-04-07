import { useEffect, useRef } from 'react';

import { Flex } from '@geti/ui';

import { SchemaEpisodeVideo } from '../../../api/openapi-spec';
import { fetchClient } from '../../../api/client';
import { useEpisodeViewer } from './episode-viewer-provider.component';

export const EpisodeVideoCell = ({ episodeVideo, datasetId }: { episodeVideo: SchemaEpisodeVideo, datasetId: string }) => {
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

    return (
        <Flex>
            <video ref={videoRef} src={url} />
        </Flex>
    );
};
