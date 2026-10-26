import { Flex, View } from '@geti-ui/ui';

import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { EpisodeTag } from '../episodes/episode-tag';
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
                <EpisodeTag indicator={nextEpisodeNumber}>
                    {state.is_recording ? (
                        <Flex alignItems='center' gap='size-50' UNSAFE_className={classes.recordingLabel}>
                            <View UNSAFE_className={classes.recordingDot} />
                            Recording
                        </Flex>
                    ) : (
                        label
                    )}
                </EpisodeTag>
            </View>
        </SidebarSection>
    );
};
