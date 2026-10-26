import { useEffect, useState } from 'react';

import { Text } from '@geti-ui/ui';

import { formatElapsed } from './format';
import { useRecording } from './recording-provider';

import classes from './recording-sidebar.module.css';

export const RecordingTimer = () => {
    const { recordingStartedAt } = useRecording();
    const [now, setNow] = useState(() => Date.now());

    useEffect(() => {
        if (recordingStartedAt === null) {
            return;
        }

        setNow(Date.now());
        const timer = window.setInterval(() => setNow(Date.now()), 1000);

        return () => window.clearInterval(timer);
    }, [recordingStartedAt]);

    const elapsed = recordingStartedAt === null ? 0 : now - recordingStartedAt;

    return <Text UNSAFE_className={classes.timer}>{formatElapsed(elapsed)}</Text>;
};
