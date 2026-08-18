export const getPathSegment = (path: string, idx: number): string => {
    const segments = path.split('/');
    return segments.length > idx ? segments.slice(0, idx + 1).join('/') : segments.join('/');
};

export const toMMSS = (timeInSeconds: number): string => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
};

const pluralRules = new Intl.PluralRules('en');

export const pluralize = (count: number, singular: string, plural: string): string =>
    pluralRules.select(count) === 'one' ? singular : plural;

export const formatDuration = (ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
};
