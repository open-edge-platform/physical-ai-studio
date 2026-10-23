import {
    ActionButton,
    AlertDialog,
    Button,
    DialogTrigger,
    Flex,
    Text,
    toast,
    Tooltip,
    TooltipTrigger,
    View,
} from '@geti-ui/ui';
import { ChevronLeft, ChevronRight, Pause, Play } from '@geti-ui/ui/icons';

import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaEpisode } from '../../../../api/openapi-spec';
import { Player } from '../use-player';
import {
    canSubmitTrim,
    clamp,
    formatSeconds,
    formatTrimRangeLabel,
    getFrameAt,
    getFullTrimRange,
    suggestTrimRange,
    TrimRange,
    withTrimEnd,
    withTrimStart,
} from './trim-range';
import { useTrimEpisodeQuery } from './use-trim-episode';

import classes from './trim-toolbar.module.css';

interface TrimToolbarProps {
    datasetId: string;
    episode: SchemaEpisode;
    player: Player;
    time: number;
    seek: (time: number) => void;
    range: TrimRange;
    setRange: (range: TrimRange) => void;
}

export const TrimToolbar = ({ datasetId, episode, player, time, seek, range, setRange }: TrimToolbarProps) => {
    const { length, fps } = episode;
    const { isPending, trimEpisode } = useTrimEpisodeQuery(datasetId, episode.episode_index);

    const currentFrame = getFrameAt(time, fps, length);
    const kept = range.end - range.start;
    const removedSeconds = (length - kept) / fps;

    const getPlayheadFrame = () => getFrameAt(player.timeRef.current, fps, length);
    const stepFrame = (step: number) => seek(clamp(getPlayheadFrame() + step, 0, length - 1) / fps);

    const handleSuggest = () => {
        const suggestion = suggestTrimRange(episode.actions, episode.action_keys, fps);
        if (suggestion === undefined) {
            toast.info('No idle frames detected.');
            return;
        }
        setRange(suggestion);
    };

    const handleTrim = async () => {
        try {
            await trimEpisode(range.start, range.end);
            toast.positive('Episode trimmed.');
        } catch (error: unknown) {
            toast.negative(getApiErrorMessage(error) ?? 'Failed to trim episode.');
        }
    };

    return (
        <Flex alignItems='center' wrap gap={'size-100'}>
            <Flex alignItems={'center'} gap='size-75'>
                <ActionButton aria-label='Previous frame' onPress={() => stepFrame(-1)}>
                    <ChevronLeft />
                </ActionButton>
                <ActionButton
                    aria-label={player.isPlaying ? 'Pause' : 'Play'}
                    onPress={player.isPlaying ? player.pause : player.play}
                >
                    {player.isPlaying ? <Pause /> : <Play />}
                </ActionButton>
                <ActionButton aria-label='Next frame' onPress={() => stepFrame(1)}>
                    <ChevronRight />
                </ActionButton>
                <Text UNSAFE_className={classes.readout}>
                    frame {currentFrame} / {length - 1} · {formatSeconds(time)}
                </Text>
            </Flex>
            <View flexGrow={1} />

            <Flex alignItems={'center'} gap='size-75'>
                <ActionButton onPress={() => setRange(withTrimStart(range, getPlayheadFrame()))}>[ In</ActionButton>
                <ActionButton onPress={() => setRange(withTrimEnd(range, getPlayheadFrame() + 1, length))}>
                    Out ]
                </ActionButton>
                <ActionButton onPress={() => setRange(getFullTrimRange(length))}>Full</ActionButton>
                <TooltipTrigger>
                    <ActionButton onPress={handleSuggest}>Suggest range</ActionButton>
                    <Tooltip>
                        Select frames where the robot is moving, skipping idle frames at the start and end.
                    </Tooltip>
                </TooltipTrigger>
            </Flex>

            <Flex alignItems={'center'} gap='size-75'>
                <Text UNSAFE_className={classes.readout} width={'size-3600'}>
                    {/* The range is half-open [start, end), so the last kept frame is end - 1. */}
                    <Text>
                        keep {range.start}–{range.end - 1} · {kept}/{length} frames · {formatSeconds(kept / fps)}
                    </Text>
                    {removedSeconds > 0 && (
                        <Text UNSAFE_className={classes.removed}> (−{formatSeconds(removedSeconds)})</Text>
                    )}
                </Text>
                <DialogTrigger>
                    <Button
                        variant='accent'
                        isPending={isPending}
                        isDisabled={!canSubmitTrim(range, length) || isPending}
                    >
                        Trim
                    </Button>
                    <AlertDialog
                        title='Trim episode'
                        variant='warning'
                        primaryActionLabel='Trim'
                        cancelLabel='Cancel'
                        isPrimaryActionDisabled={isPending}
                        onPrimaryAction={handleTrim}
                    >
                        Are you sure you want to trim this episode to {formatTrimRangeLabel(range, fps)}? Frames outside
                        this range will be removed from the episode.
                    </AlertDialog>
                </DialogTrigger>
            </Flex>
        </Flex>
    );
};
