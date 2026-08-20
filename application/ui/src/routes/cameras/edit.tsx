import { FormPreviewLayout } from '../../components/form-preview-layout';
import { CameraForm } from '../../features/robots/camera-form/form';
import { Preview } from '../../features/robots/camera-form/preview';
import { CameraFormProvider } from '../../features/robots/camera-form/provider';
import { useCamera } from '../../features/robots/use-camera';

export const Edit = () => {
    const camera = useCamera();

    return (
        <CameraFormProvider camera={camera}>
            <FormPreviewLayout
                form={<CameraForm isEdit />}
                preview={<Preview />}
                previewProps={{ backgroundColor: 'gray-50', padding: 'size-400' }}
            />
        </CameraFormProvider>
    );
};
