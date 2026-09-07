import { useEffect, useRef } from 'react';

import { fetchClient } from '../../../api/client';
import { SchemaEpisodeVideo } from '../../../api/openapi-spec';
import { useFittedMediaSize } from '../../cameras/use-fitted-media-size';
import { useEpisodeViewer } from './episode-viewer-provider.component';
import { Player } from './use-player';

type EpisodeVideoProps = {
    url: string | undefined;
    player: Player;
    episodeVideo: SchemaEpisodeVideo | undefined;
};

const EpisodeVideo = ({ url, player, episodeVideo }: EpisodeVideoProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        const video = videoRef.current;
        const start = episodeVideo?.start;
        if (!video || start === undefined || !Number.isFinite(start)) return;

        video.currentTime = player.timeRef.current + start;
        if (player.isPlaying) {
            video.play();
        } else {
            video.pause();
        }
    }, [player, episodeVideo?.start, videoRef]);

    useEffect(() => {
        const video = videoRef.current;
        const start = episodeVideo?.start;

        if (video && player.isSeeking && start !== undefined && Number.isFinite(start)) {
            const interval = setInterval(() => {
                video.currentTime = player.timeRef.current + start;
            }, 1000 / 60);
            return () => clearInterval(interval);
        }
    }, [player, episodeVideo?.start]);

    const { containerRef, width, height } = useFittedMediaSize(
        videoRef.current?.videoWidth,
        videoRef.current?.videoHeight
    );

    /* eslint-disable jsx-a11y/media-has-caption */
    return (
        <div ref={containerRef} style={{ height: '100%', width: '100%' }}>
            {url !== undefined && <video ref={videoRef} src={url} width={width} height={height} />}
        </div>
    );
};

export const EpisodeVideoCell = ({ videoId, datasetId }: { videoId: string; datasetId: string | undefined }) => {
    const { player, episode } = useEpisodeViewer();

    const episodeVideo: SchemaEpisodeVideo | undefined = episode.videos[videoId];

    const url =
        episodeVideo === undefined || datasetId === undefined
            ? undefined
            : fetchClient.PATH('/api/dataset/{dataset_id}/video/{video_path}', {
                  params: {
                      path: {
                          dataset_id: datasetId,
                          video_path: episodeVideo.path,
                      },
                  },
              });

    return <EpisodeVideo key={episode.episode_index} url={url} player={player} episodeVideo={episodeVideo} />;
};
