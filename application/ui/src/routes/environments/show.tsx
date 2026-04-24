import { Button, Flex, View } from '@geti-ui/ui';

import { useProjectId } from '../../features/projects/use-project';
import { Preview } from '../../features/robots/environment-form/preview';
import { EnvironmentForm, EnvironmentFormProvider } from '../../features/robots/environment-form/provider';
import { useEnvironment } from '../../features/robots/use-environment';
import { paths } from '../../router';

const Header = () => {
    const { project_id } = useProjectId();

    return (
        <Flex width='100%'>
            <View
                width='100%'
                borderBottomColor={'gray-400'}
                borderBottomWidth={'thin'}
                padding='size-200'
                margin={'size-200'}
                marginBottom={'size-200'}
                marginTop={'size-100'}
            >
                <Flex justifyContent={'end'} width='100%'>
                    <Button href={paths.project.datasets.index({ project_id })} variant='secondary'>
                        Record dataset
                    </Button>
                </Flex>
            </View>
        </Flex>
    );
};

export const EnvironmentShow = () => {
    const environment = useEnvironment();

    const environmentForm: EnvironmentForm = {
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
            <Header />
            <Preview />
        </EnvironmentFormProvider>
    );
};
