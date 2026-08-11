import { FormPreviewLayout } from '../../components/form-preview-layout';
import { Preview } from '../../features/robots/robot-form/preview';
import { RobotFormProvider } from '../../features/robots/robot-form/provider';
import { UpdateRobotForm } from '../../features/robots/robot-form/update-form';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { useRobot } from '../../features/robots/use-robot';

export const Edit = () => {
    const robot = useRobot();

    return (
        <RobotModelsProvider>
            <RobotFormProvider robot={robot}>
                <FormPreviewLayout
                    form={<UpdateRobotForm />}
                    preview={<Preview />}
                    previewProps={{ backgroundColor: 'gray-50', padding: 'size-400' }}
                />
            </RobotFormProvider>
        </RobotModelsProvider>
    );
};
