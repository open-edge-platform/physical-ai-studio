import { FormEvent, useState } from 'react';

import {
    Button,
    ButtonGroup,
    ComboBox,
    Flex,
    Form,
    Heading,
    Item,
    Keyboard,
    ProgressCircle,
    StatusLight,
    Text,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { formatKeyCombo, getEffectiveBindings } from '../../../features/hotkeys/key-combo';
import { useHotkey } from '../../../features/hotkeys/use-hotkey';
import { RobotControlView } from '../../../features/robots/robot-control/robot-control-view';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { useRuntimeSession } from '../../../features/robots/runtime-session-provider';
import { paths } from '../../../router';

import classes from './recording-viewer.module.css';

export const RecordingViewer = () => {
    const {
        dataset,
        state,
        startEpisode,
        discardEpisode,
        saveEpisode,
        readyForRecording,
        environment,
        observation,
        actions,
    } = useRuntimeSession();

    if (dataset === undefined) {
        throw 'Cannot load recording viewer without dataset.';
    }
    const [task, setTask] = useState<string>(dataset.default_task);

    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const bindings = getEffectiveBindings(settings.hotkeys.bindings);

    useHotkey(bindings['recording.start_episode'], () => startEpisode.mutate(task), !state.is_recording && task !== '');
    useHotkey(
        bindings['recording.accept_episode'],
        () => saveEpisode.mutate(),
        state.is_recording && !saveEpisode.isPending
    );
    useHotkey(
        bindings['recording.discard_episode'],
        () => discardEpisode.mutate(),
        state.is_recording && !saveEpisode.isPending
    );

    const onStart = (e: FormEvent) => {
        e.preventDefault();
        if (task !== '') {
            startEpisode.mutate(task);
        }
    };

    if (!readyForRecording) {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'} direction={'column'}>
                <Heading level={2}>
                    <Text>Initializing</Text>
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                </Heading>
                <Flex direction='column' margin='size-200'>
                    <StatusLight variant={state.dataset_loaded ? 'positive' : 'yellow'}>Dataset</StatusLight>
                    <StatusLight variant={state.connected ? 'positive' : 'yellow'}>Environment</StatusLight>
                </Flex>
                <Button
                    variant={'secondary'}
                    href={paths.project.datasets.show({
                        dataset_id: dataset.id!,
                        project_id: dataset.project_id,
                    })}
                >
                    Cancel
                </Button>
            </Flex>
        );
    }

    return (
        <RobotModelsProvider>
            <Flex direction={'column'} height={'100%'} position={'relative'}>
                <Form validationBehavior='native' onSubmit={onStart}>
                    <Flex justifyContent={'start'} gap='size-100' height='size-800'>
                        <ComboBox
                            isReadOnly={state.is_recording}
                            errorMessage={'A task is required in order to record.'}
                            name='Task'
                            flex
                            isRequired
                            allowsCustomValue
                            inputValue={task}
                            onInputChange={setTask}
                        >
                            <Item key={dataset.default_task}>{dataset.default_task}</Item>
                        </ComboBox>
                        {state.is_recording ? (
                            <ButtonGroup>
                                <Button
                                    isDisabled={saveEpisode.isPending}
                                    variant={'negative'}
                                    onPress={() => discardEpisode.mutate()}
                                >
                                    <Text>Discard</Text>
                                    <Keyboard UNSAFE_className={classes.hotkey}>
                                        {formatKeyCombo(bindings['recording.discard_episode'])}
                                    </Keyboard>
                                </Button>
                                <Button isPending={saveEpisode.isPending} onPress={() => saveEpisode.mutate()}>
                                    <Text>Accept</Text>
                                    <Keyboard UNSAFE_className={classes.hotkey}>
                                        {formatKeyCombo(bindings['recording.accept_episode'])}
                                    </Keyboard>
                                </Button>
                            </ButtonGroup>
                        ) : (
                            <Button type={'submit'}>
                                <Text>Start episode</Text>
                                <Keyboard UNSAFE_className={classes.hotkey}>
                                    {formatKeyCombo(bindings['recording.start_episode'])}
                                </Keyboard>
                            </Button>
                        )}
                    </Flex>
                </Form>
                <RobotControlView
                    environment={environment}
                    isReady={state.connected}
                    joints={{
                        get current() {
                            return actions.current ?? observation.current;
                        },
                    }}
                />
            </Flex>
        </RobotModelsProvider>
    );
};
