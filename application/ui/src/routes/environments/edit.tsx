import { FormPreviewLayout } from '../../components/form-preview-layout';
import { EnvironmentForm } from '../../features/robots/environment-form/form';
import { Preview } from '../../features/robots/environment-form/preview';
import {
    EnvironmentFormProvider,
    EnvironmentFormState,
    RobotConfiguration,
} from '../../features/robots/environment-form/provider';
import { UpdateEnvironmentButton } from '../../features/robots/environment-form/update-environment-button';
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
                form={<EnvironmentForm heading='Update environment' submitButton={<UpdateEnvironmentButton />} />}
                preview={<Preview />}
                previewProps={{ backgroundColor: 'gray-50' }}
            />
        </EnvironmentFormProvider>
    );
};
