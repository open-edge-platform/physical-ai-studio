import { formatDuration } from '../../../utils';

export const durationBetween = (start: string, end: string): string => {
    return formatDuration(new Date(end).getTime() - new Date(start).getTime());
};
