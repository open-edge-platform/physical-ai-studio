import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import {
    SchemaCalibrationConfig,
    SchemaCameraConfig,
    SchemaProjectConfig,
    SchemaRobotConfig,
    SchemaRobotPortInfo,
} from '../../../api/openapi-spec';

interface NewProjectContext {
    project: SchemaProjectConfig;
    setProject: Dispatch<SetStateAction<SchemaProjectConfig>>;
    updateCamera: (name: string, data: SchemaCameraConfig) => void;
    renameCamera: (oldName: string, newName: string) => void;
    availableRobots: SchemaRobotPortInfo[];
    followerCalibrations: SchemaCalibrationConfig[];
    leaderCalibrations: SchemaCalibrationConfig[];
    isValid: () => boolean;
    isRobotSetupValid: () => boolean;
    isCameraSetupValid: () => boolean;
}

function createEmptyRobot(config: Partial<SchemaRobotConfig> = {}): SchemaRobotConfig {
    return {
        serial_id: '',
        id: '',
        type: 'follower',
        ...config,
    };
}

export function createEmptyCamera(config: Partial<SchemaCameraConfig>): SchemaCameraConfig {
    return {
        id: '',
        fps: 30,
        name: '',
        type: 'OpenCV',
        width: 640,
        height: 480,
        use_depth: false,
        ...config,
    };
}

function createEmptyProject(name: string): SchemaProjectConfig {
    const id = uuidv4();
    return {
        id,
        name,
        datasets: [],
        fps: 30,
        cameras: [createEmptyCamera({ name: 'front' })],
        robots: [createEmptyRobot({ type: 'follower' }), createEmptyRobot({ type: 'leader' })],
    };
}
export const NewProjectContext = createContext<NewProjectContext | undefined>(undefined);

export function useNewProject(): NewProjectContext {
    const context = useContext(NewProjectContext);

    if (context === undefined) {
        throw new Error('No project data context');
    }

    return context;
}

export const NewProject = ({ children }: { children: ReactNode }) => {
    const [project, setProject] = useState<SchemaProjectConfig>(createEmptyProject('New Project'));
    const robotsQuery = $api.useQuery('get', '/api/hardware/robots');
    const { data: calibrations } = $api.useQuery('get', '/api/hardware/calibrations');

    const availableRobots = robotsQuery.data ?? [];
    const followerCalibrations = (calibrations ?? []).filter((m) => m.robot_type == 'robot');
    const leaderCalibrations = (calibrations ?? []).filter((m) => m.robot_type == 'teleoperator');

    const updateCamera = (name: string, data: SchemaCameraConfig) => {
        setProject({
            ...project,
            cameras: {
                ...project.cameras,
                [name]: data,
            },
        });
    };

    const renameCamera = (oldName: string, newName: string) => {
        const cameras = project.cameras.map((c) => {
            if (c.name == oldName) {
                return { ...c, name: newName };
            } else if (c.name == newName) {
                return { ...c, name: oldName };
            } else return c;
        });

        setProject({
            ...project,
            cameras,
        });
    };

    const isCameraSetupValid = () => {
        return !project.cameras.find((camera) => {
            return camera.id === '' || camera.name === '';
        });
    };

    const isRobotSetupValid = () => {
        return !project.robots.find((robot) => {
            return robot.id === '' || robot.serial_id === '';
        });
    };

    const isValid = () => {
        return isCameraSetupValid() && isRobotSetupValid();
    };

    return (
        <NewProjectContext.Provider
            value={{
                project,
                setProject,
                updateCamera,
                renameCamera,
                availableRobots,
                leaderCalibrations,
                followerCalibrations,
                isValid,
                isRobotSetupValid,
                isCameraSetupValid,
            }}
        >
            {children}
        </NewProjectContext.Provider>
    );
};
