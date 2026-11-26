// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import {
    Button,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Form,
    NumberField,
    StatusLight,
    Text,
    toast,
} from '@geti-ui/ui';

import { SchemaAckMessage, SchemaAutopilotRecordingState, SchemaSessionState } from '../../api/openapi-spec';

export type AutopilotRecordingCommand =
    | {
          command: 'start_autopilot_recording';
          dataset_id: string;
          options: { max_episodes: number; max_attempts: number | null; recovery_fraction: number | null };
      }
    | { command: 'stop_autopilot_recording' };

type Send = (command: AutopilotRecordingCommand) => Promise<SchemaAckMessage>;

const STUCK_STARTS_HELP = 'Share of episodes that start from a stuck pose, to show how to recover';
const ATTEMPTS_HELP = "Leave empty for the simulation's default";

const ENDED = new Set(['done', 'stopped', 'failed']);

/** Whether the session records automatically now (neither done, stopped nor failed). */
export const isAutopilotRecording = (recording: SchemaAutopilotRecordingState | null | undefined) =>
    recording != null && !ENDED.has(recording.phase);

const PHASES: Record<string, string> = {
    starting: 'Starting',
    preparing: 'The simulation lays out the next episode',
    recording: 'Recording',
    saving: 'Saving the episode',
    stopping: 'Stopping',
    done: 'Done',
    stopped: 'Stopped',
    failed: 'Failed',
};

const DISCARD_REASONS: Record<string, string> = {
    failure: 'failed',
    stalled: 'stalled',
    aborted: 'aborted',
    stale_stream: 'a camera stopped sending',
    too_short: 'too short',
    dataset: 'the dataset discarded it',
    save_failed: 'saving failed',
};

/** "2 failed, 1 stalled": why episodes were discarded, so a broken scene shows. */
export const describeDiscards = (discards: Record<string, number>): string =>
    Object.entries(discards)
        .filter(([, count]) => count > 0)
        .map(([reason, count]) => `${count} ${DISCARD_REASONS[reason] ?? reason}`)
        .join(', ');

/** Where the recording is, how many episodes it kept and discarded, and why the last one went as it did. */
export const AutopilotRecordingProgress = ({ recording }: { recording: SchemaAutopilotRecordingState }) => {
    const running = isAutopilotRecording(recording);
    const variant = recording.phase === 'failed' ? 'negative' : running ? 'info' : 'positive';
    const budget = recording.max_episodes > 0 ? ` of ${recording.max_episodes}` : '';
    const discards = describeDiscards(recording.discards ?? {});
    const message = running ? '' : recording.message;
    const last = recording.last;
    // A stop discards the episode it was recording, for the stop's own reason: say that once.
    const lastDiscard = last != null && !last.kept && last.reason && last.reason !== message ? last.reason : null;

    return (
        <div role='status' aria-live='polite'>
            <Flex direction='column' gap='size-75'>
                <StatusLight variant={variant}>{PHASES[recording.phase] ?? recording.phase}</StatusLight>
                {running && recording.task && <Text>Task: {recording.task}</Text>}
                <Text>
                    Saved {recording.saved}
                    {budget}, discarded {recording.discarded}
                    {discards !== '' ? ` (${discards})` : ''}
                </Text>
                {lastDiscard !== null && <Text>Last discarded: {lastDiscard}</Text>}
                {message && <Text>{message}</Text>}
            </Flex>
        </div>
    );
};

/**
 * Record a simulated follower led by its simulation's autopilot, automatically: the session asks the
 * simulation for each episode, records it with the episode's instruction as its task, and keeps the
 * successes only. Stopping holds the arm and discards an episode being recorded.
 */
export const AutomaticRecording = ({
    state,
    canRecord,
    send,
    options,
    unavailable = null,
}: {
    state: SchemaSessionState | null;
    /** A dataset loaded and the environment working. */
    canRecord: boolean;
    send: Send;
    /** The episode options the running scene takes (its simulation says); none shown while unknown. */
    options?: string[] | null;
    /** Why automatic recording cannot start now (no simulation Studio started, a scene without it); empty
     * while that is still being checked; ``null`` when it can. The session checks it again on start. */
    unavailable?: string | null;
}) => {
    const [episodes, setEpisodes] = useState(0);
    const [stuckStarts, setStuckStarts] = useState(0);
    const [attempts, setAttempts] = useState<number | null>(null);
    const [isPending, setIsPending] = useState(false);
    const recording = state?.autopilot_recording ?? null;
    const takesStuckStarts = options?.includes('recovery_fraction') ?? false;
    const takesAttempts = options?.includes('max_attempts') ?? false;
    const running = isAutopilotRecording(recording);
    const dataset = state?.dataset ?? null;
    const blocked = unavailable !== null && unavailable !== undefined;

    const run = async (command: AutopilotRecordingCommand, failure: string) => {
        setIsPending(true);
        try {
            const ack = await send(command);
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? failure);
            }
        } finally {
            setIsPending(false);
        }
    };

    const onStart = (event: FormEvent) => {
        event.preventDefault();
        if (dataset === null) {
            return;
        }
        void run(
            {
                command: 'start_autopilot_recording',
                dataset_id: dataset.dataset_id,
                options: {
                    max_episodes: episodes,
                    max_attempts: takesAttempts ? attempts : null,
                    recovery_fraction: takesStuckStarts && stuckStarts > 0 ? stuckStarts : null,
                },
            },
            'Could not start recording automatically.'
        );
    };

    return (
        <Flex direction='column' gap='size-150'>
            {recording !== null && <AutopilotRecordingProgress recording={recording} />}
            {running ? (
                <Button
                    variant='negative'
                    isPending={isPending}
                    onPress={() => void run({ command: 'stop_autopilot_recording' }, 'Could not stop recording.')}
                >
                    Stop recording
                </Button>
            ) : (
                <Form validationBehavior='native' onSubmit={onStart}>
                    <Flex direction='column' gap='size-150'>
                        <NumberField
                            label='Episodes'
                            description='Successes to keep; 0 records until you stop it'
                            minValue={0}
                            maxValue={10_000}
                            step={1}
                            value={episodes}
                            onChange={(value) => setEpisodes(Number.isNaN(value) ? 0 : value)}
                            width='100%'
                        />
                        {(takesStuckStarts || takesAttempts) && (
                            <Disclosure isQuiet>
                                <DisclosureTitle>More</DisclosureTitle>
                                <DisclosurePanel>
                                    <Flex direction='column' gap='size-150'>
                                        {takesStuckStarts && (
                                            <NumberField
                                                label='Stuck starts'
                                                description={STUCK_STARTS_HELP}
                                                minValue={0}
                                                maxValue={0.5}
                                                step={0.05}
                                                formatOptions={{ style: 'percent' }}
                                                value={stuckStarts}
                                                onChange={(value) => setStuckStarts(Number.isNaN(value) ? 0 : value)}
                                                width='100%'
                                            />
                                        )}
                                        {takesAttempts && (
                                            <NumberField
                                                label='Attempts per episode'
                                                description={ATTEMPTS_HELP}
                                                minValue={1}
                                                maxValue={5}
                                                step={1}
                                                value={attempts ?? Number.NaN}
                                                onChange={(value) => setAttempts(Number.isNaN(value) ? null : value)}
                                                width='100%'
                                            />
                                        )}
                                    </Flex>
                                </DisclosurePanel>
                            </Disclosure>
                        )}
                        {blocked && unavailable !== '' && <Text>{unavailable}</Text>}
                        <Button
                            type='submit'
                            variant='accent'
                            isPending={isPending}
                            isDisabled={!canRecord || dataset === null || blocked}
                        >
                            Record automatically
                        </Button>
                    </Flex>
                </Form>
            )}
        </Flex>
    );
};
