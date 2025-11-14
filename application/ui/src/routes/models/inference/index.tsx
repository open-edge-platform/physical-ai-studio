import { Button, View, Flex, Slider } from "@geti/ui"
import { useParams } from "react-router"
import { useInference } from "./use-inference";
import { $api } from "../../../api/client";
import { SchemaInferenceConfig, SchemaTeleoperationConfig } from "../../../api/openapi-spec";
import { useProject } from "../../../features/projects/use-project";
import { RobotViewer } from "../../../features/robots/controller/robot-viewer";
import { RobotModelsProvider } from "../../../features/robots/robot-models-context";
import { useState } from "react";
import { TELEOPERATION_CONFIG_CACHE_KEY } from "../../datasets/record/utils";

export const Index = () => {
    const { project_id, model_id } = useParams();
    const { data: model } = $api.useSuspenseQuery("get", "/api/models/{model_id}", { params: { query: { uuid: model_id! } } })

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

    let actions = observation.current === undefined ? undefined : formatActionDictToArray(observation.current.actions)

    if (calculateTrajectory.data !== undefined) {
        const i = index ?? 0;
        if (calculateTrajectory.data.length >= i) {
            actions = calculateTrajectory.data[i];
        }
    }

    return (
        <RobotModelsProvider>
            <Flex height="100%" flex direction={'column'}>
                <Flex flex>
                    <Button onPress={() => startTask(0)}>StartTask</Button>
                    <Button onPress={stop}>Pause</Button>
                    <Button onPress={disconnect}>Disconnect</Button>
                    <Button onPress={() => calculateTrajectory.mutate()}>Calculate a Trajectory</Button>
                    {calculateTrajectory.data && <Slider maxValue={calculateTrajectory.data?.length} value={index} onChange={setIndex} />}
                </Flex>

                <Flex flex={3} minWidth={0}>
                    <RobotViewer jointValues={actions}/>
                </Flex>
            </Flex>
        </RobotModelsProvider>
    )
}
