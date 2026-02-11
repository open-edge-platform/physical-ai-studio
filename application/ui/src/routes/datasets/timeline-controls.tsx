import { ActionButton, Flex, Text } from '@geti/ui';
import { Play, StepBackward, Close as Stop } from '@geti/ui/icons';

import { Player } from './use-player';
import { toMMSS } from '../../utils'

interface TimelineControlsProps {
    player: Player;
}
export const TimelineControls = ({
    player: { isPlaying, rewind, pause, play, duration, time },
}: TimelineControlsProps) => {
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
            <Text alignSelf={'center'}>
                {toMMSS(time)}/{toMMSS(duration)}
            </Text>
        </Flex>
    );
};
