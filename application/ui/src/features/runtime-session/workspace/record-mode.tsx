import { FormEvent, useEffect, useState } from 'react';

import { Button, Flex, Form, Keyboard, Slider, Text, TextArea, toast } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSessionDataset } from '../../../api/openapi-spec';
import { formatKeyCombo, getEffectiveBindings } from '../../hotkeys/key-combo';
import { useHotkey } from '../../hotkeys/use-hotkey';
import { RelativeTeleop, TeleopSwitch } from '../cells/robot-cell';
import { ERROR_TOAST, failureOf, useRuntimeSession } from '../runtime-session-provider';

import classes from './record-mode.module.css';

const RecordingTimer = () => {
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        const timer = window.setInterval(() => setSeconds((value) => value + 1), 1000);
        return () => window.clearInterval(timer);
    }, []);

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60)
        .toString()
        .padStart(2, '0');
    const remainingSeconds = (seconds % 60).toString().padStart(2, '0');

    return (
        <span className={classes.timer} aria-hidden='true'>
            {hours > 0
                ? hours.toString().padStart(2, '0') + ':' + minutes + ':' + remainingSeconds
                : minutes + ':' + remainingSeconds}
        </span>
    );
};

const RecordingStatus = () => (
    <div className={classes.toolbarStatus} role='status' aria-live='polite'>
        <span className={classes.recordingDot} aria-hidden='true' />
        <div className={classes.statusCopy}>
            <strong>Recording episode</strong>
        </div>
        <RecordingTimer />
    </div>
);

/** Send a recording command and report a failure; ``isPending`` while one is in flight. */
const useRecordingCommand = () => {
    const { send } = useRuntimeSession();
    const [isPending, setIsPending] = useState(false);

    const run = async (command: Parameters<typeof send>[0]) => {
        setIsPending(true);
        try {
            const ack = await send(command);
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? 'The recording command failed.', ERROR_TOAST);
            }
        } finally {
            setIsPending(false);
        }
    };

    return { run, isPending };
};

/** The dataset's default task, as the task box starts with it; refetched when the dataset is swapped. */
const useDefaultTask = (dataset_id: string | undefined) => {
    const { data } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: dataset_id ?? '' } } },
        { enabled: dataset_id !== undefined }
    );
    return data?.default_task ?? '';
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
    const { state, statuses } = useRuntimeSession();
    const dataset = state?.dataset ?? undefined;
    const defaultTask = useDefaultTask(dataset?.dataset_id);
    const [task, setTask] = useState(defaultTask);
    const { run, isPending } = useRecordingCommand();

    useEffect(() => setTask(defaultTask), [defaultTask]);

    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings.hotkeys.bindings);

    const isRecording = dataset?.is_recording ?? false;
    // The dataset stays loaded without a working environment, with nothing to record from.
    const canRecord =
        dataset !== undefined &&
        !isPending &&
        state?.environment != null &&
        failureOf(statuses, 'environment') === null;

    const start = () => void run({ command: 'start_episode', task });
    const save = () => void run({ command: 'save_episode' });
    const discard = () => void run({ command: 'discard_episode' });

    useHotkey(bindings['recording.start_episode'], start, canRecord && !isRecording && task !== '');
    useHotkey(bindings['recording.accept_episode'], save, canRecord && isRecording);
    useHotkey(bindings['recording.discard_episode'], discard, canRecord && isRecording);

    const onStart = (event: FormEvent) => {
        event.preventDefault();
        if (task !== '') {
            start();
        }
    };

    return (
        <Flex direction='column' gap='size-200'>
            {state?.environment?.leader_id != null && (
                <>
                    <TeleopSwitch />
                    <RelativeTeleop />
                </>
            )}
            <Form validationBehavior='native' onSubmit={onStart}>
                <Flex direction='column' gap='size-150'>
                    <TextArea
                        label='Task'
                        isReadOnly={isRecording}
                        errorMessage={'A task is required in order to record.'}
                        name='Task'
                        width='100%'
                        isRequired
                        value={task}
                        onChange={setTask}
                    />
                    {dataset !== undefined && <RecordingRate dataset={dataset} />}
                    {isRecording && <RecordingStatus />}
                    {isRecording ? (
                        <Flex gap='size-100'>
                            <Button flex isDisabled={isPending} variant={'negative'} onPress={discard}>
                                <Text>Discard</Text>
                                <Keyboard UNSAFE_className={classes.hotkey}>
                                    {formatKeyCombo(bindings['recording.discard_episode'])}
                                </Keyboard>
                            </Button>
                            <Button flex isPending={isPending} onPress={save}>
                                <Text>Accept</Text>
                                <Keyboard UNSAFE_className={classes.hotkey}>
                                    {formatKeyCombo(bindings['recording.accept_episode'])}
                                </Keyboard>
                            </Button>
                        </Flex>
                    ) : (
                        <Button type={'submit'} isPending={isPending} isDisabled={dataset === undefined}>
                            <Text>Start episode</Text>
                            <Keyboard UNSAFE_className={classes.hotkey}>
                                {formatKeyCombo(bindings['recording.start_episode'])}
                            </Keyboard>
                        </Button>
                    )}
                </Flex>
            </Form>
        </Flex>
    );
};
