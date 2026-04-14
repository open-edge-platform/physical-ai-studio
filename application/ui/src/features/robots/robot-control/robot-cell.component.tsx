import { View } from '@geti-ui/ui';

import { RobotViewer } from '../controller/robot-viewer';
import { Observation, useRobotControl } from '../robot-control-provider';
import { RobotModelsProvider } from '../robot-models-context';

const getActionObservationSource = (observation?: Observation): { [joint: string]: number } | undefined => {
    if (observation === undefined) {
        return undefined;
    }
    if (observation.actions !== null) {
        return observation.actions;
    }
    return observation.state;
};

const InnerCell = ({ robot_id }: { robot_id: string }) => {
    const { observation, environment } = useRobotControl();

    const observation_source = getActionObservationSource(observation.current);
    const action_values = observation_source === undefined ? undefined : Object.values(observation_source);
    const action_keys = observation_source === undefined ? undefined : Object.keys(observation_source);
    if (environment.robots === undefined) {
        return <></>;
    }

    return (
        <View minWidth='size-4000' minHeight='size-4000' width='100%' height='100%' backgroundColor={'gray-600'}>
            <RobotViewer
                key={robot_id}
                featureValues={action_values}
                featureNames={action_keys}
                robot={environment.robots[0].robot}
            />
        </View>
    );
};

export const RobotCell = ({ robot_id }: { robot_id: string }) => {
    return (
        <RobotModelsProvider>
            <InnerCell robot_id={robot_id} />
        </RobotModelsProvider>
    );
};
