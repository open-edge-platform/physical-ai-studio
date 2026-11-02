import { FormPreviewLayout } from '../../components/form-preview-layout';
import { Preview } from '../../features/robots/environment-form/preview';
import {
    EnvironmentFormProvider,
    EnvironmentFormState,
    RobotConfiguration,
} from '../../features/robots/environment-form/provider';
import { UpdateEnvironmentForm } from '../../features/robots/environment-form/update-form';
import { useEnvironment } from '../../features/robots/use-environment';

export const Edit = () => {
    const environment = useEnvironment();

    const environmentForm: EnvironmentFormState = {
        name: environment.name,
        cameras: environment.cameras?.map(({ id, name }) => ({ camera_id: id!, name: name! })) ?? [],
        robots:
            environment.robots?.map((robot): RobotConfiguration => {
                return {
                    robot_id: robot.robot.id,
                    teleoperator:
                        robot.tele_operator.type === 'robot'
                            ? {
                                  type: 'robot',
                                  robot_id: robot.tele_operator.robot_id,
                              }
                            : { type: 'none' },
                };
            }) ?? [],
    };

    return (
        <EnvironmentFormProvider environment={environmentForm}>
            <FormPreviewLayout
                form={<UpdateEnvironmentForm />}
                preview={<Preview />}
                previewProps={{ backgroundColor: 'gray-50' }}
            />
        </EnvironmentFormProvider>
    );
};
