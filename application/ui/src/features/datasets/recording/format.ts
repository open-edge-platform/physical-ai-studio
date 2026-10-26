const pad = (value: number): string => value.toString().padStart(2, '0');

/** Formats elapsed milliseconds as `mm:ss`, or `hh:mm:ss` from one hour onwards. */
export const formatElapsed = (ms: number): string => {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    return hours > 0 ? `${pad(hours)}:${pad(minutes)}:${pad(seconds)}` : `${pad(minutes)}:${pad(seconds)}`;
};

export const formatEpisodeCounts = (episodesInDataset: number, episodesThisSession: number): string | null => {
    if (episodesInDataset === 0) {
        return null;
    }

    const inDataset = `${episodesInDataset} ${episodesInDataset === 1 ? 'episode' : 'episodes'} in dataset`;

    return episodesThisSession === 0 ? inDataset : `${inDataset} · ${episodesThisSession} this session`;
};
