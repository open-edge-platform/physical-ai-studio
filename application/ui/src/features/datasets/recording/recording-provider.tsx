import { createContext, ReactNode, useContext, useLayoutEffect, useRef, useState } from 'react';

import { $api } from '../../../api/client';
import { HotkeyActionId } from '../../hotkeys/hotkey-actions';
import { getEffectiveBindings } from '../../hotkeys/key-combo';
import { useHotkey } from '../../hotkeys/use-hotkey';
import { useRuntimeSession } from '../../robots/runtime-session-provider';

interface RecordingContextValue {
    defaultTask: string;
    task: string;
    setTask: (task: string) => void;
    bindings: Record<HotkeyActionId, string>;
    recordingStartedAt: number | null;
}

const RecordingContext = createContext<RecordingContextValue | null>(null);

const useRecordingStartedAt = (isRecording: boolean): number | null => {
    const [recordingStartedAt, setRecordingStartedAt] = useState<number | null>(null);
    const wasRecording = useRef(isRecording);

    useLayoutEffect(() => {
        if (wasRecording.current === isRecording) {
            return;
        }

        wasRecording.current = isRecording;
        setRecordingStartedAt(isRecording ? Date.now() : null);
    }, [isRecording]);

    return recordingStartedAt;
};

export const RecordingProvider = ({ children }: { children: ReactNode }) => {
    const { dataset, state, startEpisode, saveEpisode, discardEpisode } = useRuntimeSession();

    if (dataset === undefined) {
        throw new Error('Cannot record episodes without a dataset.');
    }

    const defaultTask = dataset.default_task;
    const [customTask, setCustomTask] = useState<string | null>(null);
    const task = customTask ?? defaultTask;
    const recordingStartedAt = useRecordingStartedAt(state.is_recording);

    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings.hotkeys.bindings);

    const canStart = !state.is_recording && !startEpisode.isPending && task !== '';
    const canFinish = state.is_recording && !saveEpisode.isPending && !discardEpisode.isPending;

    useHotkey(bindings['recording.start_episode'], () => startEpisode.mutate(task), canStart);
    useHotkey(bindings['recording.accept_episode'], () => saveEpisode.mutate(undefined), canFinish);
    useHotkey(bindings['recording.discard_episode'], () => discardEpisode.mutate(undefined), canFinish);

    return (
        <RecordingContext.Provider value={{ defaultTask, task, setTask: setCustomTask, bindings, recordingStartedAt }}>
            {children}
        </RecordingContext.Provider>
    );
};

export const useRecording = (): RecordingContextValue => {
    const ctx = useContext(RecordingContext);
    if (ctx === null) throw new Error('useRecording must be used within RecordingProvider');
    return ctx;
};
