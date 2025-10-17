export const getPathSegment = (path: string, idx: number): string => {
    const segments = path.split('/');
    return segments.length > idx ? segments.slice(0, idx + 1).join('/') : segments.join('/');
};
