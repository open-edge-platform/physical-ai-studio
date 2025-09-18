import { Button, ButtonGroup, Flex, Heading, Item, TabList, TabPanels, Tabs, Text, View } from '@geti/ui';
import { Contract, FitScreen, Gear } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { CamerasView } from './cameras';
import { NewProject, useNewProject } from './new-project.provider';
import { PropertiesView } from './properties';
import { RobotsView } from './robots';

export const ProjectForm = () => {
    const { project, isValid, isCameraSetupValid, isRobotSetupValid } = useNewProject();
    const navigate = useNavigate();
    const saveMutation = $api.useMutation('put', '/api/projects');

    const save = () => {
        saveMutation
            .mutateAsync({
                body: project,
            })
            .then((project_id) => {
                navigate(paths.project.datasets.index({ project_id }));
            });
    };

    return (
        <Flex width='100%' height='100%' direction={'column'} alignItems='center'>
            <View flex={1} width={'100%'} maxWidth='1320px' paddingTop='size-400'>
                <Flex justifyContent={'space-between'}>
                    <Heading>New Project</Heading>
                    <ButtonGroup>
                        <Button variant='secondary' onPress={() => navigate(paths.projects.index.pattern)}>
                            Cancel
                        </Button>
                        <Button isDisabled={!isValid() || saveMutation.isPending} onPress={save}>
                            Save
                        </Button>
                    </ButtonGroup>
                </Flex>
                <Tabs aria-label='NewProject'>
                    <TabList>
                        <Item key='Properties' textValue='Properties'>
                            <Gear height='10px' fill={'white'} />
                            <Text>Properties</Text>
                        </Item>
                        <Item key='Robots' textValue='Robots'>
                            <Contract
                                height='10px'
                                fill={isRobotSetupValid() ? 'white' : 'var(--spectrum-semantic-notice-color-icon)'}
                            />
                            <Text>Robots</Text>
                        </Item>
                        <Item key='Cameras' textValue='Cameras'>
                            <FitScreen
                                height='10px'
                                fill={isCameraSetupValid() ? 'white' : 'var(--spectrum-semantic-notice-color-icon)'}
                            />
                            <Text>Cameras</Text>
                        </Item>
                    </TabList>
                    <TabPanels>
                        <Item key='Properties'>
                            <PropertiesView />
                        </Item>
                        <Item key='Robots'>
                            <RobotsView />
                        </Item>
                        <Item key='Cameras'>
                            <CamerasView />
                        </Item>
                    </TabPanels>
                </Tabs>
            </View>
        </Flex>
    );
};

export const NewProjectPage = () => {
    return (
        <NewProject>
            <ProjectForm />
        </NewProject>
    );
};
