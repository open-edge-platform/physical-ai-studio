import { useCallback, useEffect, useState } from 'react';

import { ActionButton, Flex, Text } from '@geti-ui/ui';
import { Play, StepBackward, Close as Stop } from '@geti-ui/ui/icons';

import { Player } from '../../features/datasets/episodes/use-player';
import { toMMSS } from '../../utils';

interface TimelineControlsProps {
    player: Player;
}
export const TimelineControls = ({
    player: { isPlaying, isSeeking, rewind, pause, play, duration, timeRef },
}: TimelineControlsProps) => {
    const [timeText, setTimeText] = useState<string>(`00:00/${toMMSS(duration)}`);

    const updateTime = useCallback(() => {
        setTimeText(`${toMMSS(timeRef.current)}/${toMMSS(duration)}`);
    }, [setTimeText, timeRef, duration]);

    useEffect(() => {
        if (isPlaying) {
            const interval = setInterval(() => {
                updateTime();
            }, 1000);
            return () => clearInterval(interval);
        }
    }, [isPlaying, updateTime]);

    useEffect(() => {
        if (isSeeking) {
            const interval = setInterval(() => {
                updateTime();
            }, 1000 / 60);
            return () => clearInterval(interval);
        }
    }, [isSeeking, updateTime]);

    return (
        <Flex direction={'row'}>
            <ActionButton aria-label='Rewind' isQuiet onPress={rewind}>
                <StepBackward fill='white' />
            </ActionButton>
            {isPlaying ? (
                <ActionButton aria-label='Pause' isQuiet onPress={pause}>
                    <Stop fill='white' />
                </ActionButton>
            ) : (
                <ActionButton aria-label='Play' isQuiet onPress={play}>
                    <Play fill='white' />
                </ActionButton>
            )}
            <Text alignSelf={'center'}>{timeText}</Text>
        </Flex>
    );
};
