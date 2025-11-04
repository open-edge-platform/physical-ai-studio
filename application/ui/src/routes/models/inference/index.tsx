import { Button, View, Flex } from "@geti/ui"
import { useParams } from "react-router"
import { useInference } from "./use-inference";
import { $api } from "../../../api/client";
import { SchemaInferenceConfig } from "../../../api/openapi-spec";
import { useProject } from "../../../features/projects/use-project";
import { RobotViewer } from "../../../features/robots/controller/robot-viewer";
import { RobotModelsProvider } from "../../../features/robots/robot-models-context";

export const Index = () => {
    const { project_id, model_id } = useParams();
    const { data: model } = $api.useSuspenseQuery("get", "/api/models/{model_id}", { params: { query: { uuid: model_id! } } })

    const project = useProject();

    const config: SchemaInferenceConfig = {
        model,
        task_index: 0,
        fps: project.config!.fps,
        cameras: [
            {
                "id": "fe99defd-d699-4bbf-99ef-3c806ef11d19",
                "port_or_device_id": "/dev/video4",
                "name": "top",
                "driver": "webcam",
                "width": 640,
                "height": 480,
                "fps": 30,
                "use_depth": false,
            },
            {
                "id": "169f3090-ec64-45da-8434-8e419f0c7f1d",
                "port_or_device_id": "/dev/video6",
                "name": "grabber",
                "driver": "webcam",
                "width": 640,
                "height": 480,
                "fps": 30,
                "use_depth": false,
            },
        ],
        robot: {
            "id": "khaos",
            "robot_type": "so101_follower",
            "serial_id": "5AA9017083",
            "port": "/dev/ttyACM2",
            "type": "follower",
        },
    }
    const { startTask, stop, state, disconnect, observation } = useInference(config);
    console.log(project_id, model_id)

    const formatActionDictToArray = (actions: { [key: string]: number }): number[] => {
        const jointNames = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
        return jointNames.map((name) => actions[`${name}.pos`]);
    };

    const actions = observation.current === undefined ? undefined : formatActionDictToArray(observation.current.actions)
    return (
        <RobotModelsProvider>
            <Flex height="100%" flex direction={'column'}>
                <Flex flex>
                    <Button onPress={() => startTask(0)}>StartTask</Button>
                    <Button onPress={stop}>Pause</Button>
                    <Button onPress={disconnect}>Disconnect</Button>
                </Flex>

                <Flex flex={3} minWidth={0}>
                    <RobotViewer jointValues={actions} />
                </Flex>
            </Flex>
        </RobotModelsProvider>
    )
}
