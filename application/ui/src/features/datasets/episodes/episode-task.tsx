import { useState, type FormEvent, type KeyboardEvent } from 'react';

import { ActionButton, Flex, Text, TextField, toast } from '@geti-ui/ui';
import { Checkmark, Close, Edit } from '@geti-ui/ui/icons';

import { getApiErrorMessage } from '../../../api/errors';
import { SchemaEpisode } from '../../../api/openapi-spec';
import { useUpdateEpisodeTask } from './use-update-episode-task';

interface EpisodeTaskProps {
    datasetId: string;
    episode: SchemaEpisode;
}

export const EpisodeTask = ({ datasetId, episode }: EpisodeTaskProps) => {
    const [draft, setDraft] = useState<string | null>(null);
    const updateTask = useUpdateEpisodeTask(datasetId, episode.episode_index);

    if (draft === null) {
        return (
            <Flex alignItems='center' gap='size-50'>
                <Text>{episode.tasks.join(', ')}</Text>
                <ActionButton isQuiet aria-label='Edit task' onPress={() => setDraft(episode.tasks[0] ?? '')}>
                    <Edit />
                </ActionButton>
            </Flex>
        );
    }

    const task = draft.trim();
    const isUnchanged = episode.tasks.length === 1 && episode.tasks[0] === task;
    const isSaveDisabled = task.length === 0 || updateTask.isPending;

    const save = async () => {
        if (isSaveDisabled) {
            return;
        }
        if (isUnchanged) {
            setDraft(null);
            return;
        }
        try {
            await updateTask.mutateAsync({
                params: { path: { dataset_id: datasetId, episode_index: episode.episode_index } },
                body: { task },
            });
            setDraft(null);
        } catch (error) {
            toast.negative(getApiErrorMessage(error) ?? 'Could not change the task.');
        }
    };

    const onSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        void save();
    };

    const onKeyDown = (event: KeyboardEvent) => {
        if (event.key === 'Escape' && !updateTask.isPending) {
            setDraft(null);
        }
    };

    return (
        <form onSubmit={onSubmit}>
            <Flex alignItems='center' gap='size-50'>
                <TextField
                    // eslint-disable-next-line jsx-a11y/no-autofocus
                    autoFocus
                    aria-label='Episode task'
                    width='size-4600'
                    value={draft}
                    onChange={setDraft}
                    onKeyDown={onKeyDown}
                    isDisabled={updateTask.isPending}
                />
                <ActionButton isQuiet aria-label='Save task' onPress={() => void save()} isDisabled={isSaveDisabled}>
                    <Checkmark />
                </ActionButton>
                <ActionButton
                    isQuiet
                    aria-label='Cancel editing task'
                    onPress={() => setDraft(null)}
                    isDisabled={updateTask.isPending}
                >
                    <Close />
                </ActionButton>
            </Flex>
        </form>
    );
};
