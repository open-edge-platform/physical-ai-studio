import { useEffect, useState } from 'react';

import { Text } from '@geti-ui/ui';

import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { formatElapsed } from './format';
import { useRecording } from './recording-provider';

import classes from './recording-sidebar.module.css';

export const RecordingTimer = () => {
    const { state } = useRuntimeSession();
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

    const label =
        state.is_recording && recordingStartedAt !== null ? formatElapsed(now - recordingStartedAt) : formatElapsed(0);

    return <Text UNSAFE_className={classes.timer}>{label}</Text>;
};
