import { CameraDriver, useCameraForm, useSetCameraForm } from '../provider';

export const useCameraFormFields = <T extends CameraDriver>(driver: T) => {
    const { getFormData } = useCameraForm();
    const { updateFormData } = useSetCameraForm();
    const formData = getFormData(driver);

    const updateField = <K extends keyof typeof formData>(field: K, value: (typeof formData)[K]) => {
        updateFormData(driver, (prev) => ({ ...prev, [field]: value }));
    };

    const updatePayload = (update: Partial<NonNullable<(typeof formData)['payload']>>) => {
        updateFormData(driver, (prev) => ({
            ...prev,
            payload: {
                ...prev.payload,
                ...update,
            },
        }));
    };

    return {
        formData,
        updateField,
        updatePayload,
    };
};
