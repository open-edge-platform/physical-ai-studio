import { Preview } from '../../features/robots/environment-form/preview';
import { EnvironmentFormProvider, EnvironmentFormState } from '../../features/robots/environment-form/provider';
import { useEnvironment } from '../../features/robots/use-environment';

export const EnvironmentShow = () => {
    const environment = useEnvironment();

    const environmentForm: EnvironmentFormState = {
        name: environment.name,
        camera_ids: environment.cameras?.map(({ id }) => id!) ?? [],
        robots:
            environment.robots?.map((robot) => {
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
            <Preview />
        </EnvironmentFormProvider>
    );
};
