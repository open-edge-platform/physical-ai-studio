import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { SchemaRobot, SchemaRobotCamera, SchemaRobotType } from '../../../api/openapi-spec';

type RobotForm = {
    name: string;
    type: SchemaRobotType | null;
    serial_id: string | null;

    cameras: Array<SchemaRobotCamera>;
};

export type RobotFormState = RobotForm | null;

export const RobotFormContext = createContext<RobotFormState>(null);
export const SetRobotFormContext = createContext<Dispatch<SetStateAction<RobotForm>> | null>(null);

export const useRobotFormBody = (robot_id: string): SchemaRobot | null => {
    const robotForm = useRobotForm();

    if (robotForm === undefined) {
        return null;
    }

    if (robotForm.type === null || robotForm.name === null || robotForm.serial_id === null) {
        return null;
    }

    return {
        id: robot_id,
        name: robotForm.name,
        type: robotForm.type,
        serial_id: robotForm.serial_id,
        cameras: robotForm.cameras
            .filter(({ fingerprint }) => fingerprint !== '')
            .map((camera) => {
                return {
                    fingerprint: camera.fingerprint,
                    name: camera.name,
                };
            }),
    };
};

export const RobotFormProvider = ({ children, robot }: { children: ReactNode; robot?: SchemaRobot }) => {
    const [value, setValue] = useState<RobotForm>({
        name: robot?.name ?? '',
        type: robot?.type ?? null,
        serial_id: robot?.serial_id ?? null,

        cameras:
            robot?.cameras?.map((camera) => {
                return {
                    name: camera.name,
                    fingerprint: camera.fingerprint ?? '',
                };
            }) ?? [],
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
