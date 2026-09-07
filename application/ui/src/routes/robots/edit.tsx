import { FormPreviewLayout } from '../../components/form-preview-layout';
import { UnavailableRobotViewer } from '../../features/robots/controller/robot-viewer';
import { Preview } from '../../features/robots/robot-form/preview';
import { RobotFormProvider } from '../../features/robots/robot-form/provider';
import { UpdateRobotForm } from '../../features/robots/robot-form/update-form';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { isUnavailableRobot } from '../../features/robots/robot-types';
import { useRobot } from '../../features/robots/use-robot';

export const Edit = () => {
    const robot = useRobot();

    if (isUnavailableRobot(robot)) {
        return <UnavailableRobotViewer robotType={robot.type} />;
    }

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
