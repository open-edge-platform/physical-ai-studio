import { Badge, Flex, Text, View } from '@geti-ui/ui';

import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { SidebarSection } from './sidebar-section';
import { useRecordingEpisodes } from './use-recording-episodes';

import classes from './recording-sidebar.module.css';

export const EpisodeSection = () => {
    const { state, saveEpisode } = useRuntimeSession();
    const { nextEpisodeNumber } = useRecordingEpisodes();

    const label = saveEpisode.isPending ? 'Saving…' : state.is_recording ? null : 'Next episode';

    return (
        <SidebarSection label='Episode'>
            <View>
                <Flex alignItems='baseline' gap='size-125'>
                    <Badge variant='yellow'>{`E${nextEpisodeNumber}`}</Badge>
                    {label && !state.is_recording && <Text UNSAFE_className={classes.episodeLabel}>{label}</Text>}
                    {state.is_recording && (
                        <Flex alignItems={'center'} gap={'size-50'} UNSAFE_className={classes.recordingLabel}>
                            <View UNSAFE_className={classes.recordingDot} />
                            <Text>Recording</Text>
                        </Flex>
                    )}
                </Flex>
            </View>
        </SidebarSection>
    );
};
