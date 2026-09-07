import { RefObject, useState } from 'react';

import { View } from '@geti-ui/ui';

import { SchemaEnvironmentWithRelations } from '../../../api/openapi-spec';
import { useInterval } from '../../../routes/datasets/use-interval';
import { RobotViewer, UnavailableRobotViewer } from '../controller/robot-viewer';
import { RobotModelsProvider } from '../robot-models-context';
import { isUnavailableRobot } from '../robot-types';

export const RobotCell = ({
    robot_id,
    environment,
    joints,
}: {
    robot_id: string;
    environment: SchemaEnvironmentWithRelations;
    joints: RefObject<Record<string, number> | undefined>;
}) => {
    const [current, setCurrent] = useState<Record<string, number>>();
    useInterval(() => {
        setCurrent(joints.current === undefined ? undefined : { ...joints.current });
    }, 1000 / 30);

    if (environment.robots === undefined) {
        return <></>;
    }

    const environmentRobot = environment.robots.find((robot) => robot.robot.id === robot_id)?.robot;
    if (environmentRobot === undefined) return <></>;

    if (isUnavailableRobot(environmentRobot)) {
        return <UnavailableRobotViewer robotType={environmentRobot.type} />;
    }

    const action_values = current === undefined ? undefined : Object.values(current);
    const action_keys = current === undefined ? undefined : Object.keys(current);

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
