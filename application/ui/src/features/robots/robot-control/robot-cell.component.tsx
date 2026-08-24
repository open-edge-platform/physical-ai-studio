import { View } from '@geti-ui/ui';

import { RobotViewer, UnavailableRobotViewer } from '../controller/robot-viewer';
import { Observation, useRobotControl } from '../robot-control-provider';
import { RobotModelsProvider } from '../robot-models-context';
import { isUnavailableRobot } from '../robot-types';

const getActionObservationSource = (observation?: Observation): { [joint: string]: number } | undefined => {
    if (observation === undefined) {
        return undefined;
    }
    if (observation.actions !== null) {
        return observation.actions;
    }
    return observation.state;
};

export const RobotCell = ({ robot_id }: { robot_id: string }) => {
    const { observation, environment } = useRobotControl();

    const observation_source = getActionObservationSource(observation.current);
    const action_values = observation_source === undefined ? undefined : Object.values(observation_source);
    const action_keys = observation_source === undefined ? undefined : Object.keys(observation_source);
    if (environment.robots === undefined) {
        return <></>;
    }

    const environmentRobot = environment.robots.find((robot) => robot.robot.id === robot_id)?.robot;
    if (environmentRobot === undefined) return <></>;

    if (isUnavailableRobot(environmentRobot)) {
        return <UnavailableRobotViewer robotType={environmentRobot.type} />;
    }

    return (
        <RobotModelsProvider>
            <View minWidth='size-4000' minHeight='size-4000' width='100%' height='100%' backgroundColor={'gray-600'}>
                <RobotViewer
                    key={robot_id}
                    featureValues={action_values}
                    featureNames={action_keys}
                    robot={environmentRobot}
                />
            </View>
        </RobotModelsProvider>
    );
};
