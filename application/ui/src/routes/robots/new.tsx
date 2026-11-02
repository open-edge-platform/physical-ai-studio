import { FormPreviewLayout } from '../../components/form-preview-layout';
import { CreateRobotForm } from '../../features/robots/robot-form/create-form';
import { Preview } from '../../features/robots/robot-form/preview';

export const New = () => {
    return <FormPreviewLayout form={<CreateRobotForm />} preview={<Preview />} />;
};
