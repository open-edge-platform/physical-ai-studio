import { ReactNode } from 'react';

import { Flex, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSessionDataset } from '../../../api/openapi-spec';
import { EpisodeTag } from '../../datasets/episodes/episode-tag';
import { PendingRecordingCommand } from './recording-command';

import classes from './record-mode.module.css';

type EpisodeTagLabel = 'saving' | 'discarding' | 'recording' | 'next';

/** What to say next to the episode number; a command in flight wins over the recording state. */
export const getEpisodeTagLabel = (pendingCommand: PendingRecordingCommand, isRecording: boolean): EpisodeTagLabel => {
    if (pendingCommand === 'save_episode') return 'saving';
    if (pendingCommand === 'discard_episode') return 'discarding';
    return isRecording ? 'recording' : 'next';
};

const EPISODE_LABEL: Record<EpisodeTagLabel, ReactNode> = {
    saving: 'Saving…',
    discarding: 'Discarding…',
    recording: (
        <Flex alignItems='center' gap='size-50' UNSAFE_className={classes.recordingLabel}>
            <View UNSAFE_className={classes.recordingDot} />
            Recording
        </Flex>
    ),
    next: 'Next episode',
};

/** The next episode's number and what is happening to it. */
export const RecordingEpisodeTag = ({
    dataset,
    pendingCommand,
}: {
    dataset: SchemaSessionDataset;
    pendingCommand: PendingRecordingCommand;
}) => {
    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: { path: { dataset_id: dataset.dataset_id } },
    });
    // Episodes saved in this session land in the dataset only when the recording is finished.
    const next = episodes.length + dataset.episodes_recorded + 1;
    const label = getEpisodeTagLabel(pendingCommand, dataset.is_recording);

    return (
        <Flex direction='column' gap='size-150' minWidth={0}>
            <Text UNSAFE_className={classes.sectionLabel}>Episode</Text>
            <View>
                <EpisodeTag indicator={next}>{EPISODE_LABEL[label]}</EpisodeTag>
            </View>
        </Flex>
    );
};
