import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { SchemaEpisode, SchemaTeleoperationConfig } from '../../api/openapi-spec';

type RecordingContextValue = null | {
    isRecording: boolean;
    setIsRecording: (config: boolean) => void;
    recordingConfig: SchemaTeleoperationConfig | undefined,
    setRecordingConfig: (config: SchemaTeleoperationConfig | undefined) => void;
};
const RecordingContext = createContext<RecordingContextValue>(null);

export const RecordingProvider = ({ children }: { children: ReactNode }) => {
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [recordingConfig, setRecordingConfig] = useState<SchemaTeleoperationConfig>();


    const setRecordingConfigProxy = (config: SchemaTeleoperationConfig | undefined) => {
      setRecordingConfig(config)
      setIsRecording(true);
    }

    return (
        <RecordingContext.Provider
            value={{
                isRecording,
                setIsRecording,
                recordingConfig,
                setRecordingConfig: setRecordingConfigProxy,
            }}
        >
            {children}
        </RecordingContext.Provider>
    );
};

export const useRecording = () => {
    return useContext(RecordingContext)!;
};

//export const useLoadModelMutation = () => {
//    const { setModels } = useRobotModels();
//
//    return useMutation({
//        mutationFn: async (path: string) => {
//            const loader = new URDFLoader();
//            return new Promise<URDFRobot>((resolve, reject) => {
//                loader.load(path, resolve, console.info, reject);
//            });
//        },
//        onSuccess: async (model) => {
//            setModels((models) => [...models, model]);
//        },
//    });
//};
