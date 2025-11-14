import { Button, ButtonGroup, Flex, Link, Heading, ComboBox, Item, ProgressCircle } from "@geti/ui"
import { useParams } from "react-router"
import { useInference } from "./use-inference";
import { $api } from "../../../api/client";
import { SchemaInferenceConfig, SchemaTeleoperationConfig } from "../../../api/openapi-spec";
import { useProject } from "../../../features/projects/use-project";
import { RobotViewer } from "../../../features/robots/controller/robot-viewer";
import { RobotModelsProvider } from "../../../features/robots/robot-models-context";
import { useState } from "react";
import { TELEOPERATION_CONFIG_CACHE_KEY } from "../../datasets/record/utils";
import { CameraView } from "../../datasets/camera-view";
import { Back, Play, Pause, StepBackward } from '@geti/ui/icons';
import { paths } from "../../../router";

const useInferenceParams = () => {
    const { project_id, model_id } = useParams();

    if (project_id === undefined) {
        throw new Error('Unknown project_id parameter');
    }

    if (model_id === undefined) {
        throw new Error('Unknown model_id parameter');
    }

    return { project_id, model_id };
}

export const Index = () => {
    const { project_id, model_id } = useInferenceParams();

    const { data: model } = $api.useSuspenseQuery("get", "/api/models/{model_id}", { params: { query: { uuid: model_id } } })
    const { data: tasks } = $api.useSuspenseQuery("get", "/api/models/{model_id}/tasks", { params: { query: { uuid: model_id } } })
    const [task, setTask] = useState<string>(tasks[0] ?? "");

    const project = useProject();
    const [index, setIndex] = useState<number>();

    const cachedConfig = JSON.parse(localStorage.getItem(TELEOPERATION_CONFIG_CACHE_KEY) ?? "{}") as SchemaTeleoperationConfig
    const config: SchemaInferenceConfig = {
        model,
        task_index: 0,
        fps: project.config!.fps,
        cameras: cachedConfig.cameras,
        robot: cachedConfig.follower,
    }
    const { startTask, stop, state, disconnect, observation, calculateTrajectory } = useInference(config);

    const formatActionDictToArray = (actions: { [key: string]: number }): number[] => {
        const jointNames = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
        return jointNames.map((name) => actions[`${name}.pos`]);
    };

    let actions = observation.current === undefined ? undefined : formatActionDictToArray(observation.current.state)

    if (calculateTrajectory.data !== undefined) {
        const i = index ?? 0;
        if (calculateTrajectory.data.length >= i) {
            actions = calculateTrajectory.data[i];
        }
    }
    if (!state.initialized) {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'}>
                <Heading>Initializing</Heading>
                <ProgressCircle isIndeterminate />
            </Flex>
        );
    }

    const onStart = () => {
        startTask(tasks.indexOf(task))
    }

    return (
        <RobotModelsProvider>
            <Flex flex direction={'column'} height={'100%'} position={'relative'}>
                <Flex alignItems={'center'} gap='size-100' height="size-400" margin="size-200">
                    <Link aria-label='Rewind' href={paths.project.models.index({ project_id })}>
                        <Back fill='white' />
                    </Link>
                    <Heading>Model Run {model.name}</Heading>
                    <ComboBox
                        flex
                        isRequired
                        allowsCustomValue={false}
                        inputValue={task}
                        onInputChange={setTask}
                    >
                        {tasks.map((task, index) => (
                            <Item key={index}>{task}</Item>
                        ))}
                    </ComboBox>
                    <ButtonGroup>
                        <Button variant='primary'>
                            <StepBackward fill='white' />
                            Restart
                        </Button>

                        { state.is_running
                            ? <Button variant='primary' onPress={stop}>
                                <Pause fill='white' />
                                Stop
                            </Button>
                            : <Button variant='primary' onPress={onStart}>
                                <Play fill='white' />
                                Play
                            </Button>
                        }
                        <Button variant="negative">
                            Start Recording
                        </Button>
                    </ButtonGroup>
                </Flex>
                <Flex direction={'row'} flex gap={'size-100'} margin='size-200'>
                    <Flex direction={'column'} alignContent={'start'} flex gap={'size-30'}>
                        {config.cameras.map((camera) => (
                            <CameraView key={camera.id} camera={camera} observation={observation} />
                        ))}
                    </Flex>
                    <Flex flex={3} minWidth={0}>
                        <RobotViewer jointValues={actions} />
                    </Flex>
                </Flex>
            </Flex>
        </RobotModelsProvider>
    )
}
