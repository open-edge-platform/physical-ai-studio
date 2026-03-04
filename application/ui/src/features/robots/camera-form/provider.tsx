import { createContext, ReactNode, SetStateAction, useContext, useState } from 'react';

import { $api } from '../../../api/client';
import {
    SchemaBaslerCameraInput,
    SchemaGenicamCameraInput,
    SchemaIpCameraInput,
    SchemaRealsenseCameraInput,
    SchemaUsbCameraInput,
} from '../../../api/openapi-spec';
import { SchemaProjectCamera } from '../../../api/types';
import { initialBaslerState, validateBasler } from './drivers/basler';
import { initialGenicamState, validateGenicam } from './drivers/genicam';
import { initialIpCamState, validateIpCam } from './drivers/ipcam';
import { initialRealsenseState, validateRealsense } from './drivers/realsense';
import { initialUsbCameraState, validateUsbCamera } from './drivers/usb-camera';

// Utility type to make all properties (including nested) optional
type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type CameraDriver = SchemaProjectCamera['driver']; //'usb_camera' | 'ipcam' | 'basler' | 'realsense' | 'genicam';

type DriverSchemaMap = {
    usb_camera: SchemaUsbCameraInput;
    ipcam: SchemaIpCameraInput;
    basler: SchemaBaslerCameraInput;
    realsense: SchemaRealsenseCameraInput;
    genicam: SchemaGenicamCameraInput;
};

export type DriverFormSchema<K extends CameraDriver> = DeepPartial<DriverSchemaMap[K]> & {
    driver: K;
};

export const isValid = (schema: DriverFormSchema<CameraDriver>): schema is SchemaProjectCamera => {
    switch (schema.driver) {
        case 'usb_camera':
            return validateUsbCamera(schema);
        case 'ipcam':
            return validateIpCam(schema);
        case 'realsense':
            return validateRealsense(schema);
        case 'basler':
            return validateBasler(schema);
        case 'genicam':
            return validateGenicam(schema);
        default:
            return false;
    }
};

interface CameraFormState {
    activeDriver: CameraDriver;
    formData: {
        [K in CameraDriver]: DriverFormSchema<K>;
    };
}

interface CameraFormContextType {
    state: CameraFormState;
    getFormData: <T extends CameraDriver>(driver: T) => DriverFormSchema<T>;
}
interface SetCameraFormContextType {
    setActiveDriver: (driver: CameraDriver) => void;
    updateFormData: <T extends CameraDriver>(driver: T, update: SetStateAction<DriverFormSchema<T>>) => void;
}

const CameraFormContext = createContext<CameraFormContextType | null>(null);
export const SetCameraFormContext = createContext<SetCameraFormContextType | null>(null);

const getInitialCameraFormState = (camera?: SchemaProjectCamera): CameraFormState => {
    const formData = {
        usb_camera: initialUsbCameraState,
        basler: initialBaslerState,
        genicam: initialGenicamState,
        ipcam: initialIpCamState,
        realsense: initialRealsenseState,
    } satisfies CameraFormState['formData'];

    if (camera === undefined) {
        return {
            activeDriver: 'usb_camera',
            formData,
        };
    }

    return {
        activeDriver: camera.driver,
        formData: {
            ...formData,
            [camera.driver]: camera,
        },
    };
};

export function CameraFormProvider({ children, camera }: { children: ReactNode; camera?: SchemaProjectCamera }) {
    const [state, setState] = useState<CameraFormState>(() => getInitialCameraFormState(camera));

    const value: CameraFormContextType = {
        state,
        getFormData: <T extends CameraDriver>(driver: T): DriverFormSchema<T> => {
            return (state.formData[driver] || {}) as DriverFormSchema<T>;
        },
    };

    const setValue = {
        setActiveDriver: (driver: CameraDriver) => {
            setState((prev) => ({ ...prev, activeDriver: driver }));
        },

        updateFormData: <T extends CameraDriver>(driver: T, update: SetStateAction<DriverFormSchema<T>>) => {
            setState((prev) => {
                const current = (prev.formData[driver] || {}) as DriverFormSchema<T>;
                const next = typeof update === 'function' ? update(current) : update;
                return {
                    ...prev,
                    formData: {
                        ...prev.formData,
                        [driver]: next,
                    },
                };
            });
        },
    };

    return (
        <CameraFormContext.Provider value={value}>
            <SetCameraFormContext.Provider value={setValue}>{children}</SetCameraFormContext.Provider>
        </CameraFormContext.Provider>
    );
}

export const useCameraForm = () => {
    const ctx = useContext(CameraFormContext);
    if (!ctx) throw new Error('useCameraForm must be used within CameraFormProvider');
    return ctx;
};

export const useSetCameraForm = () => {
    const context = useContext(SetCameraFormContext);

    if (context === null) {
        throw new Error('useSetCameraForm was used outside of CameraFormProvider');
    }

    return context;
};

export const useCameraFormBody = (camera_id: string): SchemaProjectCamera | null => {
    const availableCamerasQuery = $api.useQuery('get', '/api/hardware/cameras');

    const { getFormData, state } = useCameraForm();

    const cameraForm = getFormData(state.activeDriver);

    if (!isValid(cameraForm)) {
        return null;
    }

    const hardware = availableCamerasQuery.data?.find((h) => h.fingerprint === cameraForm.fingerprint);

    return {
        ...cameraForm,
        hardware_name: hardware?.name ?? cameraForm.hardware_name ?? cameraForm.name,
        id: camera_id,
    };
};
