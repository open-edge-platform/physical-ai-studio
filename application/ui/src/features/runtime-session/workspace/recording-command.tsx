import { createContext, ReactNode, useCallback, useContext, useMemo, useState } from 'react';

import { toast } from '@geti-ui/ui';

import { ERROR_TOAST, useRuntimeSession } from '../runtime-session-provider';

type RecordingCommand = Parameters<ReturnType<typeof useRuntimeSession>['send']>[0];
export type PendingRecordingCommand = RecordingCommand['command'] | null;

interface RecordingCommandValue {
    run: (command: RecordingCommand) => Promise<void>;
    pendingCommand: PendingRecordingCommand;
    isPending: boolean;
}

const RecordingCommandContext = createContext<RecordingCommandValue | null>(null);

export const RecordingCommandProvider = ({ children }: { children: ReactNode }) => {
    const { send } = useRuntimeSession();
    const [pendingCommand, setPendingCommand] = useState<PendingRecordingCommand>(null);

    const run = useCallback(
        async (command: RecordingCommand) => {
            setPendingCommand(command.command);
            try {
                const ack = await send(command);
                if (!ack.ok) {
                    toast.negative(ack.error?.message ?? 'The recording command failed.', ERROR_TOAST);
                }
            } finally {
                setPendingCommand(null);
            }
        },
        [send]
    );

    const recordingValue = useMemo(
        () => ({
            run,
            pendingCommand,
            isPending: pendingCommand !== null,
        }),
        [run, pendingCommand]
    );

    return <RecordingCommandContext value={recordingValue}>{children}</RecordingCommandContext>;
};

export const useRecordingCommand = (): RecordingCommandValue => {
    const context = useContext(RecordingCommandContext);
    if (context === null) {
        throw new Error('useRecordingCommand must be used within a RecordingCommandProvider');
    }
    return context;
};
