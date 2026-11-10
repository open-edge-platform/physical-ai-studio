import { FormEvent, useEffect, useState } from 'react';

import { Button, Flex, Form, Keyboard, Slider, Text, TextArea } from '@geti-ui/ui';
import { Checkmark, Delete } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { SchemaSessionDataset } from '../../../api/openapi-spec';
import { formatKeyCombo, getEffectiveBindings } from '../../hotkeys/key-combo';
import { useHotkey } from '../../hotkeys/use-hotkey';
import { TeleopSwitch } from '../cells/robot-cell';
import { failureOf, useRuntimeSession } from '../runtime-session-provider';
import { useRecordingCommand } from './recording-command';

import classes from './record-mode.module.css';

/** Elapsed time as `mm:ss`, or `hh:mm:ss` from one hour. */
const formatElapsed = (ms: number) => {
    const total = Math.max(0, Math.floor(ms / 1000));
    const pad = (value: number) => value.toString().padStart(2, '0');
    const hours = Math.floor(total / 3600);
    const minutes = pad(Math.floor((total % 3600) / 60));
    const seconds = pad(total % 60);
    return hours > 0 ? `${pad(hours)}:${minutes}:${seconds}` : `${minutes}:${seconds}`;
};

/** Time since the recording began, from the provider so it survives a collapsed sidebar. */
const RecordingTimer = () => {
    const { recordingStartedAt } = useRuntimeSession();
    const [now, setNow] = useState(() => Date.now());

    useEffect(() => {
        if (recordingStartedAt === null) return;
        setNow(Date.now());
        const timer = window.setInterval(() => setNow(Date.now()), 1000);
        return () => window.clearInterval(timer);
    }, [recordingStartedAt]);

    return (
        <Text UNSAFE_className={classes.timer}>
            {formatElapsed(recordingStartedAt === null ? 0 : now - recordingStartedAt)}
        </Text>
    );
};

/** The dataset's default task, as the task box starts with it; refetched when the dataset is swapped. */
const useDefaultTask = (dataset_id: string | undefined) => {
    const { data } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: dataset_id ?? '' } } },
        { enabled: dataset_id !== undefined }
    );
    return data?.default_task ?? null;
};

/** The rate a dataset with nothing recorded yet is recorded at; changing it loads the dataset again at that rate. */
const RecordingRate = ({ dataset }: { dataset: SchemaSessionDataset }) => {
    const [fps, setFps] = useState(dataset.fps);
    const { run, isPending } = useRecordingCommand();

    useEffect(() => setFps(dataset.fps), [dataset.fps]);

    // Once it has episodes, here or on disk, the dataset has its own rate.
    const isFixed = dataset.fps_fixed || dataset.episodes_recorded > 0 || dataset.is_recording;
    const reload = (value: number) => {
        if (value !== dataset.fps) {
            void run({ command: 'load_dataset', dataset_id: dataset.dataset_id, fps: value });
        }
    };

    return (
        <Slider
            label='Recording rate'
            value={fps}
            onChange={setFps}
            onChangeEnd={reload}
            minValue={10}
            maxValue={100}
            step={5}
            getValueLabel={(value) => `${value} fps`}
            isDisabled={isFixed || isPending}
            width='100%'
        />
    );
};

/** Record mode's side bar: teleoperation, the task, and start / accept / discard with their hotkeys. */
export const RecordMode = () => {
    const { state, statuses, error, isLoading } = useRuntimeSession();
    const dataset = state?.dataset ?? undefined;
    const defaultTask = useDefaultTask(dataset?.dataset_id);
    const [previousDefaultTask, setPreviousDefaultTask] = useState<string | null>(defaultTask);
    const [task, setTask] = useState(defaultTask);
    const { run, isPending, pendingCommand } = useRecordingCommand();

    if (previousDefaultTask !== defaultTask && task === null) {
        setTask(defaultTask);
        setPreviousDefaultTask(defaultTask);
    }

    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings.hotkeys.bindings);

    const isRecording = dataset?.is_recording ?? false;
    // The dataset stays loaded without a working session, with nothing to record from.
    const isSessionUsable =
        dataset !== undefined &&
        state?.environment != null &&
        failureOf(statuses, 'environment') === null &&
        failureOf(statuses, 'dataset') === null &&
        error === null &&
        !isLoading;
    // Whether each command may be issued, ignoring a command in flight: the buttons show that as pending.
    const canStart = isSessionUsable && task !== null && task.trim().length > 0;
    // Finishing an episode stays possible after an error, so a recording is never stuck.
    const canFinish = dataset !== undefined;
    // The command a button or a hotkey may issue now: no other one is in flight.
    const isStartDisabled = !canStart || isPending;
    const isSaveDisabled = !canFinish || (isPending && pendingCommand !== 'save_episode');
    const isDiscardDisabled = !canFinish || (isPending && pendingCommand !== 'discard_episode');

    const start = () => {
        if (isStartDisabled || task === null) return;
        void run({ command: 'start_episode', task });
    };
    const save = () => void run({ command: 'save_episode' });
    const discard = () => void run({ command: 'discard_episode' });

    useHotkey(bindings['recording.start_episode'], start, !isStartDisabled && !isRecording);
    useHotkey(bindings['recording.accept_episode'], save, !isSaveDisabled && !isPending && isRecording);
    useHotkey(bindings['recording.discard_episode'], discard, !isDiscardDisabled && !isPending && isRecording);

    const onStart = (event: FormEvent) => {
        event.preventDefault();
        start();
    };

    const startKey = formatKeyCombo(bindings['recording.start_episode']);
    const saveKey = formatKeyCombo(bindings['recording.accept_episode']);
    const discardKey = formatKeyCombo(bindings['recording.discard_episode']);

    const timerRow = (
        <Flex alignItems='center' justifyContent='space-between' gap='size-150'>
            <Flex alignItems='center' gap='size-150'>
                <div role='status' aria-live='polite'>
                    <RecordingTimer />
                </div>
                <Keyboard UNSAFE_className={classes.hotkey}>{startKey}</Keyboard>
            </Flex>
            {dataset !== undefined && dataset.episodes_recorded > 0 && (
                <Text UNSAFE_className={classes.helper}>{dataset.episodes_recorded} this session</Text>
            )}
        </Flex>
    );

    return (
        <Flex direction='column' gap='size-300'>
            {state?.environment?.leader_id != null && <TeleopSwitch />}
            <Form validationBehavior='native' onSubmit={onStart}>
                <Flex direction='column' gap='size-300'>
                    <TextArea
                        label='Task'
                        isReadOnly={isRecording}
                        errorMessage={'A task is required in order to record.'}
                        name='Task'
                        width='100%'
                        isRequired
                        value={task ?? ''}
                        onChange={setTask}
                    />
                    {dataset !== undefined && <RecordingRate dataset={dataset} />}
                    <Flex direction='column' gap='size-150'>
                        {isRecording ? (
                            <>
                                {timerRow}
                                <Flex wrap gap='size-100'>
                                    <Button
                                        variant='accent'
                                        isPending={pendingCommand === 'save_episode'}
                                        isDisabled={isSaveDisabled}
                                        onPress={save}
                                    >
                                        <Checkmark />
                                        <Text>Save episode</Text>
                                    </Button>
                                    <Button
                                        variant='secondary'
                                        isPending={pendingCommand === 'discard_episode'}
                                        isDisabled={isDiscardDisabled}
                                        onPress={discard}
                                    >
                                        <Delete />
                                        <Text>Discard</Text>
                                    </Button>
                                </Flex>
                            </>
                        ) : (
                            <>
                                <Button
                                    type='submit'
                                    variant='accent'
                                    width='100%'
                                    isPending={isPending}
                                    isDisabled={!canStart}
                                >
                                    <Text>Start episode</Text>
                                </Button>
                                {timerRow}
                            </>
                        )}
                        <Text UNSAFE_className={classes.helper}>
                            {isRecording ? '' : 'After the episode: '}Save ({saveKey}) or Discard ({discardKey}).
                        </Text>
                    </Flex>
                </Flex>
            </Form>
        </Flex>
    );
};
