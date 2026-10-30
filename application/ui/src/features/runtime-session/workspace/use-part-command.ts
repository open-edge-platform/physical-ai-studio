import { useState } from 'react';

import { toast } from '@geti-ui/ui';

import { SchemaIncompatiblePart } from '../../../api/openapi-spec';
import { ERROR_TOAST, useRuntimeSession } from '../runtime-session-provider';

type Send = ReturnType<typeof useRuntimeSession>['send'];
export type SessionCommand = Parameters<Send>[0];

/** A load that would unload incompatible parts, waiting for the user to confirm. */
export interface PendingConfirmation {
    parts: SchemaIncompatiblePart[];
    confirm: () => void;
    cancel: () => void;
}

/**
 * Send a part command (load, reload, swap, unload) and report a failure.
 *
 * A load the session refuses with ``incompatible_parts`` becomes a ``confirmation`` to show;
 * confirming sends it again with ``unload_incompatible``.
 */
export const usePartCommand = () => {
    const { send } = useRuntimeSession();
    const [isPending, setIsPending] = useState(false);
    const [confirmation, setConfirmation] = useState<PendingConfirmation | null>(null);

    const run = async (command: SessionCommand): Promise<boolean> => {
        setIsPending(true);
        try {
            const ack = await send(command);
            if (ack.ok) {
                return true;
            }
            if (ack.error?.error_code === 'incompatible_parts' && command.command === 'load_environment') {
                setConfirmation({
                    parts: ack.error.parts ?? [],
                    confirm: () => {
                        setConfirmation(null);
                        void run({ ...command, unload_incompatible: true });
                    },
                    cancel: () => setConfirmation(null),
                });
                return false;
            }
            toast.negative(ack.error?.message ?? 'The command failed.', ERROR_TOAST);
            return false;
        } finally {
            setIsPending(false);
        }
    };

    return { run, isPending, confirmation };
};
