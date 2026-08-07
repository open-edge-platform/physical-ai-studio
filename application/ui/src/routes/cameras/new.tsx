import { FormPreviewLayout } from '../../components/form-preview-layout';
import { CameraForm } from '../../features/robots/camera-form/form';
import { Preview } from '../../features/robots/camera-form/preview';
import { CameraFormProvider } from '../../features/robots/camera-form/provider';

export const New = () => {
    return (
        <CameraFormProvider>
            <FormPreviewLayout
                form={<CameraForm />}
                preview={<Preview />}
                previewProps={{ backgroundColor: 'gray-50', padding: 'size-400' }}
            />
        </CameraFormProvider>
    );
};
