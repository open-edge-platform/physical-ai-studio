import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { $api } from '../../../api/client';
import { SchemaCameraInput, SchemaRobotCamera } from '../../../api/openapi-spec';

type CameraForm = {
    name: SchemaCameraInput['name'];
    fingerprint: SchemaCameraInput['fingerprint'] | null;
    resolution_fps: SchemaCameraInput['resolution_fps'] | null;
    resolution_width: SchemaCameraInput['resolution_width'] | null;
    resolution_height: SchemaCameraInput['resolution_height'] | null;
};

export type CameraFormState = CameraForm | null;

export const CameraFormContext = createContext<CameraFormState>(null);
export const SetCameraFormContext = createContext<Dispatch<SetStateAction<CameraForm>> | null>(null);

export const useCameraFormBody = (camera_id: string): SchemaCameraInput | null => {
    const availableCamerasQuery = $api.useQuery('get', '/api/hardware/cameras');

    const cameraForm = useCameraForm();
    const hardwareCamera = availableCamerasQuery.data?.find((camera) => {
        return cameraForm.fingerprint === camera.fingerprint;
    });

    if (cameraForm === undefined) {
        return null;
    }

    if (
        cameraForm.fingerprint === null ||
        cameraForm.name === null ||
        cameraForm.resolution_fps === null ||
        cameraForm.resolution_width === null ||
        cameraForm.resolution_height === null
    ) {
        return null;
    }

    return {
        id: camera_id,
        name: cameraForm.name,
        fingerprint: cameraForm.fingerprint,
        // The discovery endpoint needs to be backward compatible for now
        driver: hardwareCamera?.driver === 'webcam' ? 'usb_camera' : (hardwareCamera?.driver ?? ''),
        hardware_name: hardwareCamera?.name ?? '',

        payload: {
            fps: cameraForm.resolution_fps,
            width: cameraForm.resolution_width,
            height: cameraForm.resolution_height,
        },
    };
};

export const CameraFormProvider = ({ children, camera }: { children: ReactNode; camera?: SchemaRobotCamera }) => {
    const [value, setValue] = useState<CameraForm>({
        name: camera?.name ?? '',
        fingerprint: camera?.fingerprint ?? null,

        resolution_fps: camera?.resolution_fps ?? null,
        resolution_width: camera?.resolution_width ?? null,
        resolution_height: camera?.resolution_height ?? null,
    });

    return (
        <CameraFormContext.Provider value={value}>
            <SetCameraFormContext.Provider value={setValue}>{children}</SetCameraFormContext.Provider>
        </CameraFormContext.Provider>
    );
};

export const useCameraForm = () => {
    const context = useContext(CameraFormContext);

    if (context === null) {
        throw new Error('useCameraForm was used outside of CameraFormProvider');
    }

    return context;
};

export const useSetCameraForm = () => {
    const context = useContext(SetCameraFormContext);

    if (context === null) {
        throw new Error('useSetCameraForm was used outside of CameraFormProvider');
    }

    return context;
};
