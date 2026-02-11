export const getPathSegment = (path: string, idx: number): string => {
    const segments = path.split('/');
    return segments.length > idx ? segments.slice(0, idx + 1).join('/') : segments.join('/');
};

export const toMMSS = (timeInSeconds: number): string => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
};
