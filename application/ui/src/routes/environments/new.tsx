import { FormPreviewLayout } from '../../components/form-preview-layout';
import { EnvironmentForm } from '../../features/robots/environment-form/form';
import { Preview } from '../../features/robots/environment-form/preview';
import { EnvironmentFormProvider } from '../../features/robots/environment-form/provider';

export const New = () => {
    return (
        <EnvironmentFormProvider>
            <FormPreviewLayout
                form={<EnvironmentForm />}
                preview={<Preview />}
                previewProps={{ backgroundColor: 'gray-50' }}
            />
        </EnvironmentFormProvider>
    );
};
