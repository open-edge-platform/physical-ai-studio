import { FormPreviewLayout } from '../../components/form-preview-layout';
import { CreateEnvironmentForm } from '../../features/robots/environment-form/create-form';
import { Preview } from '../../features/robots/environment-form/preview';
import { EnvironmentFormProvider } from '../../features/robots/environment-form/provider';

export const New = () => {
    return (
        <EnvironmentFormProvider>
            <FormPreviewLayout
                form={<CreateEnvironmentForm />}
                preview={<Preview />}
                previewProps={{ backgroundColor: 'gray-50' }}
            />
        </EnvironmentFormProvider>
    );
};
