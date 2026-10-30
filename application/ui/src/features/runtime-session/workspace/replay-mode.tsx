import { useEffect, useState } from 'react';

import { Button, Flex, Item, Picker, ProgressBar, Text, toast } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaEpisodeInfo, SchemaReplayControlConfig } from '../../../api/openapi-spec';
import { ERROR_TOAST, failureOf, useRuntimeSession } from '../runtime-session-provider';

/** How long the follower takes to reach an episode's first frame before it plays. */
const GRACE_S = 3;

const seconds = (episode: SchemaEpisodeInfo) => episode.length / episode.fps;

/** Seconds since ``playing`` last changed to an episode, ticking while one plays. */
const useElapsed = (playing: number | null) => {
    const [elapsed, setElapsed] = useState(0);

    useEffect(() => {
        setElapsed(0);
        if (playing === null) {
            return;
        }
        const startedAt = performance.now();
        const timer = window.setInterval(() => setElapsed((performance.now() - startedAt) / 1000), 100);
        return () => window.clearInterval(timer);
    }, [playing]);

    return elapsed;
};

/** Replay mode's side bar: play one of the dataset's episodes on the follower, to look at what was recorded. */
export const ReplayMode = () => {
    const { state, statuses, send } = useRuntimeSession();
    const dataset = state?.dataset ?? undefined;

    if (dataset === undefined) {
        return null; // the workspace says what is missing
    }
    return (
        <EpisodePlayer
            key={dataset.dataset_id}
            dataset_id={dataset.dataset_id}
            unfinished={dataset.episodes_recorded}
            // An open episode would record the replay.
            canPlay={state?.environment != null && failureOf(statuses, 'environment') === null && !dataset.is_recording}
            playing={state?.control?.type === 'replay' ? state.control.episode_index : null}
            send={send}
        />
    );
};

const EpisodePlayer = ({
    dataset_id,
    unfinished,
    canPlay,
    playing,
    send,
}: {
    dataset_id: string;
    /** Saved this session: still in the recording, not yet in the dataset. */
    unfinished: number;
    canPlay: boolean;
    playing: number | null;
    send: ReturnType<typeof useRuntimeSession>['send'];
}) => {
    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: { path: { dataset_id } },
    });
    const [selected, setSelected] = useState<number | null>(episodes[0]?.episode_index ?? null);
    const [isPending, setIsPending] = useState(false);
    const elapsed = useElapsed(playing);

    const setControl = async (control: SchemaReplayControlConfig | null) => {
        setIsPending(true);
        try {
            const ack = await send({ command: 'set_control', control });
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? 'Could not replay the episode.', ERROR_TOAST);
            }
        } finally {
            setIsPending(false);
        }
    };

    const played = episodes.find((episode) => episode.episode_index === playing);
    // The grace move comes before the first frame, so the bar starts once it is over.
    const progress = played !== undefined ? Math.max(elapsed - GRACE_S, 0) / seconds(played) : 0;

    return (
        <Flex direction='column' gap='size-200'>
            <Picker
                label='Episode'
                width='100%'
                items={episodes}
                selectedKey={selected === null ? null : String(selected)}
                onSelectionChange={(key) => key !== null && setSelected(Number(key))}
                isDisabled={playing !== null}
            >
                {(episode) => (
                    <Item key={String(episode.episode_index)} textValue={`Episode ${episode.episode_index}`}>
                        {`Episode ${episode.episode_index} · ${seconds(episode).toFixed(1)} s`}
                    </Item>
                )}
            </Picker>
            {unfinished > 0 && (
                <Text>
                    {unfinished === 1 ? 'The episode' : `The ${unfinished} episodes`} recorded this session can be
                    replayed once the recording is finished.
                </Text>
            )}
            {played !== undefined && (
                <ProgressBar
                    width='100%'
                    label={elapsed < GRACE_S ? 'Moving to the first frame' : `Episode ${played.episode_index}`}
                    valueLabel={progress >= 1 ? 'Done, holding the last frame' : undefined}
                    value={Math.min(progress, 1) * 100}
                />
            )}
            {playing === null ? (
                <Button
                    isPending={isPending}
                    isDisabled={!canPlay || selected === null}
                    onPress={() =>
                        selected !== null &&
                        void setControl({ type: 'replay', episode_index: selected, grace_s: GRACE_S })
                    }
                >
                    Play
                </Button>
            ) : (
                <Button variant='secondary' isPending={isPending} onPress={() => void setControl(null)}>
                    Stop
                </Button>
            )}
        </Flex>
    );
};
