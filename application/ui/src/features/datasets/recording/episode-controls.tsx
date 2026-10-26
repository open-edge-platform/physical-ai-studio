import { Button, Flex, Keyboard, Text, View } from '@geti-ui/ui';
import { Checkmark, Delete } from '@geti-ui/ui/icons';

import { formatKeyCombo } from '../../hotkeys/key-combo';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { useRecording } from './recording-provider';
import { RecordingTimer } from './recording-timer';
import { SidebarSection } from './sidebar-section';

import classes from './recording-sidebar.module.css';

export const EpisodeControls = () => {
    const { state, startEpisode, saveEpisode } = useRuntimeSession();
    const {
        task,
        startEpisode: startRecording,
        saveEpisode: saveRecording,
        discardEpisode: discardRecording,
        bindings,
    } = useRecording();

    const startKey = formatKeyCombo(bindings['recording.start_episode']);
    const saveKey = formatKeyCombo(bindings['recording.accept_episode']);
    const discardKey = formatKeyCombo(bindings['recording.discard_episode']);

    const canStart = !state.is_recording && !startEpisode.isPending && task !== '';

    const timerRow = (
        <Flex alignItems='center' gap='size-150'>
            <RecordingTimer />
            <Keyboard UNSAFE_className={classes.hotkey}>{startKey}</Keyboard>
        </Flex>
    );

    return (
        <SidebarSection>
            {state.is_recording ? (
                <>
                    {timerRow}
                    <Flex wrap gap='size-100'>
                        <Button variant='accent' isPending={saveEpisode.isPending} onPress={saveRecording}>
                            <Checkmark />
                            <Text>Save episode</Text>
                        </Button>
                        <Button variant='secondary' isDisabled={saveEpisode.isPending} onPress={discardRecording}>
                            <Delete />
                            <Text>Discard</Text>
                        </Button>
                    </Flex>
                </>
            ) : (
                <>
                    <Button
                        variant='accent'
                        width='100%'
                        isPending={startEpisode.isPending}
                        isDisabled={!canStart && !startEpisode.isPending}
                        onPress={startRecording}
                    >
                        <Text>Start episode</Text>
                    </Button>
                    {timerRow}
                </>
            )}

            <Text UNSAFE_className={classes.helper}>
                After the episode: Save ({saveKey}) or Discard ({discardKey}).
            </Text>
        </SidebarSection>
    );
};
