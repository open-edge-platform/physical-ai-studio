import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { useMutation } from '@tanstack/react-query';
import URDFLoader, { URDFRobot } from 'urdf-loader';

type Model = URDFRobot;

type ActionContextValue = null | {
    models: Array<Model>;
    setModels: Dispatch<SetStateAction<Array<Model>>>;
};
const ActionContext = createContext<ActionContextValue>(null);

export const ActionProvider = ({ children }: { children: ReactNode }) => {
    const [models, setModels] = useState<Array<Model>>([]);

    return (
        <ActionContext.Provider
            value={{
                models,
                setModels,
            }}
        >
            {children}
        </ActionContext.Provider>
    );
};

export const useAction = () => {
    return useContext(ActionContext)!;
};

export const useLoadModelMutation = () => {
    const { setModels } = useAction();

    return useMutation({
        mutationFn: async (path: string) => {
            const loader = new URDFLoader();
            return new Promise<URDFRobot>((resolve, reject) => {
                loader.load(path, resolve, console.log, reject);
            });
        },
        onSuccess: async (model) => {
            setModels((models) => [...models, model]);
        },
    });
};
