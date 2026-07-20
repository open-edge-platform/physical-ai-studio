import { useEffect, useState } from 'react';

import { formatDuration } from '../routes/models/utils';

export const elapsedSince = (dateString: string): string => {
    const normalized = /Z|[+-]\d\d:\d\d$/.test(dateString) ? dateString : `${dateString}Z`;

    return formatDuration(new Date().getTime() - new Date(normalized).getTime());
};

export const ElapsedDuration = ({ date }: { date: string }) => {
    const [elapsed, setElapsed] = useState(() => elapsedSince(date));

    useEffect(() => {
        const updateElapsed = () => setElapsed(elapsedSince(date));
        updateElapsed();

        const intervalId = window.setInterval(updateElapsed, 1000);

        return () => window.clearInterval(intervalId);
    }, [date]);

    return elapsed;
};
