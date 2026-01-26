import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { SchemaEnvironmentInput } from '../../../api/openapi-spec';

export type RobotConfiguration = {
    robot_id: string;
    teleoperator: { type: 'robot'; robot_id: string } | { type: 'none' };
};

export type EnvironmentForm = {
    name: string;
    robots: Array<RobotConfiguration>;
    camera_ids: Array<string>;
};

export type EnvironmentFormState = EnvironmentForm | null;

export const EnvironmentFormContext = createContext<EnvironmentFormState>(null);
export const SetEnvironmentFormContext = createContext<Dispatch<SetStateAction<EnvironmentForm>> | null>(null);

export const useEnvironmentFormBody = (environment_id: string): SchemaEnvironmentInput | null => {
    const environmentForm = useEnvironmentForm();

    if (environmentForm === undefined) {
        return null;
    }

    return {
        id: environment_id,
        name: environmentForm.name,
        camera_ids: environmentForm.camera_ids,
        robots: environmentForm.robots.map((robot) => {
            return {
                robot_id: robot.robot_id,
                tele_operator:
                    robot.teleoperator.type === 'robot'
                        ? {
                              type: 'robot',
                              robot_id: robot.teleoperator.robot_id,
                          }
                        : { type: 'none' },
            };
        }),
    };
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
            camera_ids: [],
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
