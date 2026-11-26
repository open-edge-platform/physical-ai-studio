import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import {
    SchemaEnvironmentInput,
    SchemaTeleoperatorAutopilotWithRobot,
    SchemaTeleoperatorNoneWithRobot,
    SchemaTeleoperatorRobotWithRobot,
} from '../../../api/openapi-spec';

/** Who drives a follower: a leader arm, the simulation's autopilot (simulated followers only), or nobody. */
export type Teleoperator = { type: 'robot'; robot_id: string } | { type: 'autopilot' } | { type: 'none' };

/** A saved environment's teleoperator, without the robot it comes with. */
export const toTeleoperator = (
    tele_operator:
        SchemaTeleoperatorRobotWithRobot | SchemaTeleoperatorAutopilotWithRobot | SchemaTeleoperatorNoneWithRobot
): Teleoperator =>
    tele_operator.type === 'robot' ? { type: 'robot', robot_id: tele_operator.robot_id } : { type: tele_operator.type };

export type RobotConfiguration = {
    robot_id: string;
    teleoperator: Teleoperator;
};

export type CameraConfiguration = {
    camera_id: string;
};

export type EnvironmentForm = {
    name: string;
    robots: Array<RobotConfiguration>;
    cameras: Array<CameraConfiguration>;
};

export type EnvironmentFormState = EnvironmentForm | null;

export const EnvironmentFormContext = createContext<EnvironmentFormState>(null);
export const SetEnvironmentFormContext = createContext<Dispatch<SetStateAction<EnvironmentForm>> | null>(null);

export const useEnvironmentFormBody = (environment_id: string) => {
    const environmentForm = useEnvironmentForm();

    return {
        id: environment_id,
        name: environmentForm.name.trim(),
        cameras: environmentForm.cameras.map((camera) => {
            return {
                camera_id: camera.camera_id,
            };
        }),
        robots: environmentForm.robots.map((robot) => {
            return {
                robot_id: robot.robot_id,
                tele_operator: robot.teleoperator,
            };
        }),
    } satisfies SchemaEnvironmentInput;
};

export const EnvironmentFormProvider = ({
    children,
    environment,
}: {
    children: ReactNode;
    environment?: EnvironmentForm;
}) => {
    const [value, setValue] = useState<EnvironmentForm>(
        environment ?? {
            name: '',
            robots: [],
            cameras: [],
        }
    );

    return (
        <EnvironmentFormContext.Provider value={value}>
            <SetEnvironmentFormContext.Provider value={setValue}>{children}</SetEnvironmentFormContext.Provider>
        </EnvironmentFormContext.Provider>
    );
};

export const useEnvironmentForm = () => {
    const context = useContext(EnvironmentFormContext);

    if (context === null) {
        throw new Error('useEnvironmentForm was used outside of EnvironmentFormProvider');
    }

    return context;
};

export const useSetEnvironmentForm = () => {
    const context = useContext(SetEnvironmentFormContext);

    if (context === null) {
        throw new Error('useSetEnvironmentForm was used outside of EnvironmentFormProvider');
    }

    return context;
};
