import { useMemo } from 'react';

import { ComboBox, Item } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { useRecording } from './recording-provider';

const useTaskOptions = (): string[] => {
    const { dataset } = useRuntimeSession();
    const { defaultTask } = useRecording();

    if (dataset?.id === undefined) {
        throw new Error('Cannot load task options without a dataset.');
    }

    const { data: episodes } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: { path: { dataset_id: dataset.id } },
    });

    return useMemo(
        () => Array.from(new Set([defaultTask, ...episodes.flatMap((episode) => episode.tasks)])),
        [defaultTask, episodes]
    );
};

export const TaskPicker = () => {
    const { state, saveEpisode } = useRuntimeSession();
    const { task, setTask } = useRecording();
    const taskOptions = useTaskOptions();
    return (
        <ComboBox
            label='Task'
            width='100%'
            isRequired
            allowsCustomValue
            isReadOnly={state.is_recording || saveEpisode.isPending}
            validationState={task === '' ? 'invalid' : undefined}
            errorMessage='A task is required in order to record.'
            items={taskOptions.map((option) => ({ id: option, name: option }))}
            inputValue={task}
            onInputChange={setTask}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </ComboBox>
    );
};
