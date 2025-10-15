import { Play, Close as Stop, StepBackward } from '@geti/ui/icons';
import { Player } from "./use-player"
import { Flex, Text, ActionButton } from "@geti/ui"

const toMMSS = (timeInSeconds: number): string => {
    const minutes = Math.floor(timeInSeconds / 60)
    const seconds = Math.floor(timeInSeconds % 60)
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

interface TimelineControlsProps {
    player: Player
}
export const TimelineControls = ({player: {isPlaying, rewind, pause, play, duration, time}}: TimelineControlsProps) => {
    return (
        <Flex direction={'row'}>
            <ActionButton aria-label='Rewind' isQuiet onPress={rewind}>
                <StepBackward fill='white' />
            </ActionButton>
            {isPlaying
                ? <ActionButton aria-label='Pause' isQuiet onPress={pause}>
                    <Stop fill='white' />
                </ActionButton>
                : <ActionButton aria-label='Play' isQuiet onPress={play}>
                    <Play fill='white' />
                </ActionButton>
            }
            <Text alignSelf={'center'}>{toMMSS(time)}/{toMMSS(duration)}</Text>
        </Flex>
    )
}

