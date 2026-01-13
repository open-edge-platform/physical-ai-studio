import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { $api } from '../../../api/client';
import { SchemaProjectCamera } from '../../../api/types';

export type CameraFormProps = Partial<Exclude<SchemaProjectCamera, 'id' | 'driver' | 'hardware_name'>>;
export type CameraFormState = CameraFormProps | null;

export const CameraFormContext = createContext<CameraFormState>(null);
export const SetCameraFormContext = createContext<Dispatch<SetStateAction<CameraFormProps>> | null>(null);

export const useCameraFormBody = (camera_id: string): SchemaProjectCamera | null => {
    const availableCamerasQuery = $api.useQuery('get', '/api/hardware/cameras');

    const cameraForm = useCameraForm();
    const hardwareCamera = availableCamerasQuery.data?.find((camera) => {
        return cameraForm.fingerprint === camera.fingerprint;
    });

    if (cameraForm === undefined) {
        return null;
    }

    if (
        cameraForm.fingerprint === undefined ||
        cameraForm.name === undefined ||
        cameraForm.payload?.fps === undefined ||
        cameraForm.payload?.width === undefined ||
        cameraForm.payload?.height === undefined
    ) {
        return null;
    }

    return {
        id: camera_id,
        name: cameraForm.name,
        fingerprint: cameraForm.fingerprint,
        // @ts-expect-error The discovery endpoint needs to be backward compatible for now
        driver: hardwareCamera?.driver === 'webcam' ? 'usb_camera' : (hardwareCamera?.driver ?? ''),
        hardware_name: hardwareCamera?.name ?? '',

        payload: {
            fps: cameraForm.payload.fps,
            width: cameraForm.payload.width,
            height: cameraForm.payload.height,
        },
    };
};

export const CameraFormProvider = ({ children, camera }: { children: ReactNode; camera?: SchemaProjectCamera }) => {
    const [value, setValue] = useState<CameraFormProps>({
        name: camera?.name ?? '',
        fingerprint: camera?.fingerprint ?? '',
        payload: camera?.payload,
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
