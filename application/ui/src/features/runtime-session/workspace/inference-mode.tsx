import { useEffect, useState } from 'react';

import { Button, Flex, Item, Picker, Slider, Text, TextArea, toast } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaExecutionConfig, SchemaModelControlConfig } from '../../../api/openapi-spec';
import { ERROR_TOAST, failureOf, useRuntimeSession } from '../runtime-session-provider';

const EXECUTION_MODES: { key: SchemaExecutionConfig['mode']; label: string }[] = [
    { key: 'sync', label: 'Synchronous' },
    { key: 'async', label: 'Asynchronous' },
    { key: 'rtc', label: 'Real-time chunking (RTC)' },
];

// The settings the RTC trials on the reBot ran with: guide 25 of a 50-action chunk, weight 5.
const RTC_HORIZON = 25;
const RTC_GUIDANCE = 5;
// The API's own bound on execution_horizon (ExecutionConfig).
const API_MAX_HORIZON = 100;

/** The default task of the dataset the loaded model was trained on, or none. */
const useDefaultTask = (model_id: string | undefined) => {
    const { data: model } = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        { params: { path: { model_id: model_id ?? '' } } },
        { enabled: model_id !== undefined }
    );
    const dataset_id = model?.model.dataset_id ?? undefined;
    const { data: dataset } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: dataset_id ?? '' } } },
        { enabled: dataset_id !== undefined }
    );
    return dataset?.default_task ?? '';
};

/** Inference mode's side bar: the task, the action rate and how chunks are requested, then play / stop. */
export const InferenceMode = () => {
    const { state, statuses, send } = useRuntimeSession();
    const defaultTask = useDefaultTask(state?.model?.model_id);
    const [task, setTask] = useState(defaultTask);
    const [mode, setMode] = useState<SchemaExecutionConfig['mode']>('sync');
    const [threshold, setThreshold] = useState(0.5);
    // An export guides at most half its chunk; the default is the reBot trials' 25 when the chunk has room.
    const maxHorizon = Math.min(state?.model?.rtc_max_horizon ?? RTC_HORIZON, API_MAX_HORIZON);
    const [horizon, setHorizon] = useState(Math.min(RTC_HORIZON, maxHorizon));
    const [guidance, setGuidance] = useState(RTC_GUIDANCE);
    // In ms; 0 lets RTC measure the delay on the first inferences instead.
    const [latencyMs, setLatencyMs] = useState(0);
    const modelTakesRtc = state?.model?.rtc === true;
    // shortcut: set by hand, upgrade to the model's training fps once exports record it.
    const [hz, setHz] = useState(30);
    const [isPending, setIsPending] = useState(false);

    useEffect(() => setTask(defaultTask), [defaultTask]);
    useEffect(() => setHorizon(Math.min(RTC_HORIZON, maxHorizon)), [maxHorizon]);

    // A model trained on a fixed set of tasks only takes one of them.
    const tasks = state?.model?.tasks ?? [];
    const selectedTask = tasks.length > 0 && !tasks.includes(task) ? tasks[0] : task;

    const isRunning = state?.control?.type === 'model';
    const canRun =
        state?.environment != null &&
        failureOf(statuses, 'environment') === null &&
        state.model != null &&
        failureOf(statuses, 'model') === null;

    const setControl = async (control: SchemaModelControlConfig | null) => {
        setIsPending(true);
        try {
            const ack = await send({ command: 'set_control', control });
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? 'Could not run the model.', ERROR_TOAST);
            }
        } finally {
            setIsPending(false);
        }
    };

    const play = () =>
        void setControl({
            type: 'model',
            task: selectedTask,
            hz,
            // Each mode reads its own settings: the threshold for sync and async, the rest for RTC.
            execution: {
                mode,
                request_threshold: threshold,
                execution_horizon: Math.min(horizon, maxHorizon),
                max_guidance_weight: guidance,
                initial_latency_s: latencyMs > 0 ? latencyMs / 1000 : null,
            },
            // The follower eases onto the model's actions, which may start far from where it is.
            grace_s: 3,
        });

    return (
        <Flex direction='column' gap='size-200'>
            {tasks.length > 0 ? (
                <Picker
                    label='Task'
                    width='100%'
                    items={tasks.map((value) => ({ id: value }))}
                    selectedKey={selectedTask}
                    onSelectionChange={(key) => key !== null && setTask(String(key))}
                    isDisabled={isRunning}
                >
                    {(item) => <Item key={item.id}>{item.id}</Item>}
                </Picker>
            ) : (
                <TextArea label='Task' width='100%' value={task} onChange={setTask} isReadOnly={isRunning} />
            )}
            <Picker
                label='Execution'
                width='100%'
                items={EXECUTION_MODES}
                selectedKey={mode}
                onSelectionChange={(key) => key !== null && setMode(key as SchemaExecutionConfig['mode'])}
                isDisabled={isRunning}
            >
                {(item) => <Item key={item.key}>{item.label}</Item>}
            </Picker>
            {mode === 'rtc' && state?.model != null && !modelTakesRtc && (
                <Text>This model was exported without RTC inputs. Load an RTC export, or choose another mode.</Text>
            )}
            <Slider
                label='Action rate'
                width='100%'
                minValue={5}
                maxValue={100}
                step={5}
                value={hz}
                onChange={setHz}
                getValueLabel={(value) => `${value} Hz`}
                isDisabled={isRunning}
            />
            {mode === 'rtc' ? (
                <>
                    <Slider
                        label='Guided actions (horizon)'
                        width='100%'
                        minValue={1}
                        maxValue={maxHorizon}
                        step={1}
                        value={horizon}
                        onChange={setHorizon}
                        isDisabled={isRunning}
                    />
                    <Slider
                        label='Guidance weight'
                        width='100%'
                        minValue={0}
                        maxValue={10}
                        step={0.5}
                        value={guidance}
                        onChange={setGuidance}
                        isDisabled={isRunning}
                    />
                    <Slider
                        label='Expected inference latency'
                        width='100%'
                        minValue={0}
                        maxValue={1000}
                        step={10}
                        value={latencyMs}
                        onChange={setLatencyMs}
                        getValueLabel={(value) => (value > 0 ? `${value} ms` : 'Measure')}
                        isDisabled={isRunning}
                    />
                </>
            ) : (
                <Slider
                    label='Request the next chunk at'
                    width='100%'
                    minValue={0}
                    maxValue={1}
                    step={0.05}
                    value={threshold}
                    onChange={setThreshold}
                    formatOptions={{ style: 'percent' }}
                    isDisabled={isRunning}
                />
            )}
            {isRunning ? (
                <Button variant='secondary' isPending={isPending} onPress={() => void setControl(null)}>
                    Stop
                </Button>
            ) : (
                <Button isPending={isPending} isDisabled={!canRun || (mode === 'rtc' && !modelTakesRtc)} onPress={play}>
                    Play
                </Button>
            )}
        </Flex>
    );
};
