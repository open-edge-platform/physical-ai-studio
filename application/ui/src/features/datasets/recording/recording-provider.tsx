import { createContext, ReactNode, useContext, useState } from 'react';

import { $api } from '../../../api/client';
import { HotkeyActionId } from '../../hotkeys/hotkey-actions';
import { getEffectiveBindings } from '../../hotkeys/key-combo';
import { useHotkey } from '../../hotkeys/use-hotkey';
import { useRuntimeSession } from '../../robots/runtime-session-provider';

interface RecordingContextValue {
    defaultTask: string;
    task: string;
    setTask: (task: string) => void;
    recordingStartedAt: number | null;
    startEpisode: () => void;
    saveEpisode: () => void;
    discardEpisode: () => void;
    bindings: Record<HotkeyActionId, string>;
}

const RecordingContext = createContext<RecordingContextValue | null>(null);

export const RecordingProvider = ({ children }: { children: ReactNode }) => {
    const { dataset, state, startEpisode: startEpisodeMutation, saveEpisode, discardEpisode } = useRuntimeSession();

    if (dataset === undefined) {
        throw new Error('Cannot record episodes without a dataset.');
    }

    const defaultTask = dataset.default_task;
    const [customTask, setCustomTask] = useState<string | null>(null);
    const task = customTask ?? defaultTask;
    const [recordingStartedAt, setRecordingStartedAt] = useState<number | null>(null);

    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings.hotkeys.bindings);

    const startEpisode = () => {
        startEpisodeMutation.mutate(task, {
            onSuccess: () => setRecordingStartedAt(Date.now()),
            onError: () => setRecordingStartedAt(null),
        });
    };

    const saveEpisodeAndResetTimer = () => {
        saveEpisode.mutate(undefined, { onSuccess: () => setRecordingStartedAt(null) });
    };

    const discardEpisodeAndResetTimer = () => {
        discardEpisode.mutate(undefined, { onSuccess: () => setRecordingStartedAt(null) });
    };

    const canStart = !state.is_recording && !startEpisodeMutation.isPending && task !== '';
    const canFinish = state.is_recording && !saveEpisode.isPending;

    useHotkey(bindings['recording.start_episode'], startEpisode, canStart);
    useHotkey(bindings['recording.accept_episode'], saveEpisodeAndResetTimer, canFinish);
    useHotkey(bindings['recording.discard_episode'], discardEpisodeAndResetTimer, canFinish);

    return (
        <RecordingContext.Provider
            value={{
                defaultTask,
                task,
                setTask: setCustomTask,
                recordingStartedAt,
                startEpisode,
                saveEpisode: saveEpisodeAndResetTimer,
                discardEpisode: discardEpisodeAndResetTimer,
                bindings,
            }}
        >
            {children}
        </RecordingContext.Provider>
    );
};

export const useRecording = (): RecordingContextValue => {
    const ctx = useContext(RecordingContext);
    if (ctx === null) throw new Error('useRecording must be used within RecordingProvider');
    return ctx;
};
