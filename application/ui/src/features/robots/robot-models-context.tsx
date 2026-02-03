import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { useMutation } from '@tanstack/react-query';
import URDFLoader, { URDFRobot } from 'urdf-loader';

type Model = URDFRobot;

type RobotModelsContextValue = null | {
    models: Array<Model>;
    setModels: Dispatch<SetStateAction<Array<Model>>>;
};
const RobotModelsContext = createContext<RobotModelsContextValue>(null);

export const RobotModelsProvider = ({ children }: { children: ReactNode }) => {
    const [models, setModels] = useState<Array<Model>>([]);

    return (
        <RobotModelsContext.Provider
            value={{
                models,
                setModels,
            }}
        >
            {children}
        </RobotModelsContext.Provider>
    );
};

export const useRobotModels = () => {
    return useContext(RobotModelsContext)!;
};

export const useLoadModelMutation = () => {
    const { setModels } = useRobotModels();

    return useMutation({
        mutationFn: async (path: string) => {
            const loader = new URDFLoader();

            loader.packages = {
                trossen_arm_description: '/widowx',
            };

            return new Promise<URDFRobot>((resolve, reject) => {
                loader.load(path, resolve, console.info, reject);
            });
        },
        onSuccess: async (model) => {
            setModels((models) => [...models, model]);
        },
    });
};
