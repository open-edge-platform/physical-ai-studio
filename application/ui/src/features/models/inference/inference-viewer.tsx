import { useState } from 'react';

import {
    AlertDialog,
    Button,
    ButtonGroup,
    ComboBox,
    DialogContainer,
    Flex,
    Heading,
    Item,
    Link,
    ProgressCircle,
    StatusLight,
    Switch,
    Text,
} from '@geti-ui/ui';
import { Back, DownloadIcon, Pause, Play } from '@geti-ui/ui/icons';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { useProjectId } from '../../projects/use-project';
import { RobotControlView } from '../../robots/robot-control/robot-control-view';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { runtimeExportUrl } from '../runtime-export';

interface InferenceViewerProps {
    tasks: string[];
}

export const InferenceViewer = ({ tasks }: InferenceViewerProps) => {
    const { project_id } = useProjectId();

    // The prompt is free text; the dataset's tasks are offered as suggestions. Custom values must
    // stay allowed: otherwise the combo box discards typed text when it loses focus.
    const [task, setTask] = useState<string>(tasks[0] ?? '');
    const [isEmptyPromptDialogOpen, setIsEmptyPromptDialogOpen] = useState(false);
    const [recordEpisodes, setRecordEpisodes] = useState(false);

    const {
        model,
        readyForInference,
        state,
        startTask,
        stopTask,
        setFollowerSource,
        environment,
        observation,
        inferenceDevice,
        dataset,
        loadDataset,
        startEpisode,
        saveEpisode,
        discardEpisode,
    } = useRuntimeSession();

    const createDataset = $api.useMutation('post', '/api/dataset', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}', { params: { path: { project_id } } }]],
        },
    });

    // Record into a fresh dataset of this environment so training datasets stay untouched.
    const toggleRecording = (enabled: boolean) => {
        setRecordEpisodes(enabled);
        if (!enabled || dataset !== undefined) {
            return;
        }
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-');
        createDataset.mutate(
            {
                body: {
                    id: uuidv4(),
                    name: `${model?.name ?? 'model'}-inference-${timestamp}`,
                    project_id,
                    environment_id: environment.id,
                    default_task: task,
                },
            },
            {
                onSuccess: (created) => loadDataset.mutate(created),
                onError: () => setRecordEpisodes(false),
            }
        );
    };

    const canTeleoperate = state.has_leader;
    const isTeleoperating = state.follower_source === 'teleop';
    const isPolicyRunning = state.follower_source === 'policy';
    const hasOpenEpisode = state.is_recording && !isPolicyRunning;

    const play = () => {
        if (recordEpisodes && state.dataset_loaded && !state.is_recording) {
            startEpisode.mutate(task, { onSuccess: () => startTask.mutate(task) });
            return;
        }
        startTask.mutate(task);
    };

    const stop = () => {
        const wasRecording = state.is_recording;
        stopTask.mutate(undefined, {
            onSuccess: () => {
                if (wasRecording) {
                    saveEpisode.mutate();
                }
            },
        });
    };

    const exportUrl =
        model?.id !== undefined && inferenceDevice !== undefined
            ? runtimeExportUrl({
                  modelId: model.id,
                  environmentId: environment.id,
                  backend: inferenceDevice.backend,
                  device: inferenceDevice.device,
                  task,
              })
            : undefined;

    if (!readyForInference) {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'} direction={'column'}>
                <Heading level={2}>
                    <Text>Initializing</Text>
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                </Heading>
                <Flex direction='column' margin='size-200'>
                    <StatusLight variant={state.model_loaded ? 'positive' : 'yellow'}>Model</StatusLight>
                    <StatusLight variant={state.connected ? 'positive' : 'yellow'}>Environment</StatusLight>
                </Flex>
                <Button variant={'secondary'} href={paths.project.models.index({ project_id })}>
                    Cancel
                </Button>
            </Flex>
        );
    }

    return (
        <RobotModelsProvider>
            <Flex flex direction={'column'} height={'100%'} position={'relative'}>
                <Flex alignItems={'center'} gap='size-100' height='size-400' margin='size-200'>
                    <Link aria-label='Rewind' href={paths.project.models.index({ project_id })}>
                        <Back fill='white' />
                    </Link>
                    <Heading>Model Run {model?.name}</Heading>
                    <ComboBox flex aria-label='Task prompt' allowsCustomValue inputValue={task} onInputChange={setTask}>
                        {tasks.map((taskText) => (
                            <Item key={taskText}>{taskText}</Item>
                        ))}
                    </ComboBox>
                    <Switch
                        isEmphasized
                        isSelected={isTeleoperating}
                        isDisabled={
                            !canTeleoperate || setFollowerSource.isPending || startTask.isPending || stopTask.isPending
                        }
                        onChange={(enabled) => setFollowerSource.mutate(enabled ? 'teleop' : 'hold')}
                    >
                        Teleoperate
                    </Switch>
                    <Switch
                        isSelected={recordEpisodes}
                        isDisabled={
                            state.is_recording || isPolicyRunning || createDataset.isPending || loadDataset.isPending
                        }
                        onChange={toggleRecording}
                    >
                        Record episodes
                    </Switch>
                    {recordEpisodes && state.dataset_loaded && dataset !== undefined && (
                        <StatusLight variant={state.is_recording ? 'negative' : 'neutral'}>
                            {dataset.name}: {state.is_recording ? 'recording' : `${state.episodes_recorded} saved`}
                        </StatusLight>
                    )}
                    <ButtonGroup>
                        {exportUrl !== undefined && (
                            <Button
                                href={exportUrl}
                                aria-label='Download runtime export'
                                variant='secondary'
                                target='_blank'
                                rel='noopener noreferrer'
                            >
                                <DownloadIcon />
                                Runtime export
                            </Button>
                        )}
                        {hasOpenEpisode && (
                            <>
                                <Button
                                    variant='negative'
                                    isDisabled={saveEpisode.isPending}
                                    onPress={() => discardEpisode.mutate()}
                                >
                                    Discard episode
                                </Button>
                                <Button
                                    variant='secondary'
                                    isPending={saveEpisode.isPending}
                                    onPress={() => saveEpisode.mutate()}
                                >
                                    Save episode
                                </Button>
                            </>
                        )}
                        {isPolicyRunning ? (
                            <Button variant='primary' isPending={stopTask.isPending} onPress={stop}>
                                <Pause fill='white' />
                                Stop
                            </Button>
                        ) : (
                            <Button
                                variant='primary'
                                isPending={startTask.isPending || startEpisode.isPending}
                                onPress={() => (task.trim() === '' ? setIsEmptyPromptDialogOpen(true) : play())}
                            >
                                <Play fill='white' />
                                Play
                            </Button>
                        )}
                    </ButtonGroup>
                </Flex>
                <RobotControlView environment={environment} isReady={state.connected} joints={observation} />
            </Flex>
            <DialogContainer onDismiss={() => setIsEmptyPromptDialogOpen(false)}>
                {isEmptyPromptDialogOpen && (
                    <AlertDialog
                        title='Start without a task prompt?'
                        variant='warning'
                        primaryActionLabel='Start anyway'
                        secondaryActionLabel='Cancel'
                        onPrimaryAction={() => {
                            setIsEmptyPromptDialogOpen(false);
                            play();
                        }}
                        onSecondaryAction={() => setIsEmptyPromptDialogOpen(false)}
                    >
                        <Text>
                            The task prompt is empty. Policies that follow language instructions, such as Pi0.5, will
                            run without an instruction.
                        </Text>
                    </AlertDialog>
                )}
            </DialogContainer>
        </RobotModelsProvider>
    );
};
