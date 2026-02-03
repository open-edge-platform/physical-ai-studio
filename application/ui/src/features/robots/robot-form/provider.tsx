import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { SchemaRobot, SchemaRobotType } from '../../../api/openapi-spec';

type RobotForm = {
    name: string;
    type: SchemaRobotType | null;
    connection_string: string;
    serial_number: string;
};

export type RobotFormState = RobotForm | null;

export const RobotFormContext = createContext<RobotFormState>(null);
export const SetRobotFormContext = createContext<Dispatch<SetStateAction<RobotForm>> | null>(null);

export const useRobotFormBody = (robot_id: string): SchemaRobot | null => {
    const robotForm = useRobotForm();

    if (robotForm === undefined) {
        return null;
    }

    if (
        robotForm.type === null ||
        robotForm.name === null ||
        (robotForm.connection_string === null && robotForm.serial_number === null)
    ) {
        return null;
    }

    return {
        id: robot_id,
        name: robotForm.name,
        type: robotForm.type,
        connection_string: robotForm.connection_string ?? '',
        serial_number: robotForm.serial_number ?? '',
    };
};

export const RobotFormProvider = ({ children, robot }: { children: ReactNode; robot?: SchemaRobot }) => {
    const [value, setValue] = useState<RobotForm>({
        name: robot?.name ?? '',
        type: robot?.type ?? null,
        connection_string: robot?.connection_string ?? '',
        serial_number: robot?.serial_number ?? '',
    });

    return (
        <RobotFormContext.Provider value={value}>
            <SetRobotFormContext.Provider value={setValue}>{children}</SetRobotFormContext.Provider>
        </RobotFormContext.Provider>
    );
};

export const useRobotForm = () => {
    const context = useContext(RobotFormContext);

    if (context === null) {
        throw new Error('useRobotForm was used outside of RobotFormProvider');
    }

    return context;
};

export const useSetRobotForm = () => {
    const context = useContext(SetRobotFormContext);

    if (context === null) {
        throw new Error('useSetRobotForm was used outside of RobotFormProvider');
    }

    return context;
};
