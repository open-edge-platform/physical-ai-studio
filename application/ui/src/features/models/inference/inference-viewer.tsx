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
    } = useRuntimeSession();

    const canTeleoperate = state.has_leader;
    const isTeleoperating = state.follower_source === 'teleop';

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
                        {state.follower_source === 'policy' ? (
                            <Button variant='primary' isPending={stopTask.isPending} onPress={() => stopTask.mutate()}>
                                <Pause fill='white' />
                                Stop
                            </Button>
                        ) : (
                            <Button
                                variant='primary'
                                isPending={startTask.isPending}
                                onPress={() =>
                                    task.trim() === '' ? setIsEmptyPromptDialogOpen(true) : startTask.mutate(task)
                                }
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
                            startTask.mutate(task);
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
