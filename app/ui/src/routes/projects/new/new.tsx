import { Flex, Text, ButtonGroup, Button, Heading, Tabs, TabList, Item, TabPanels, View, Form, TextField, NumberField } from '@geti/ui';
import { createProjectDataContext, ProjectDataContext, useProjectDataContext } from './project-config.provider';
import { CamerasView } from './cameras';
import { RobotsView } from './robots';
import { PropertiesView } from './properties';
import { Gear, Contract, FitScreen } from '@geti/ui/icons';
import { $api } from '../../../api/client';
import { useNavigate } from 'react-router';
import { paths } from '../../../router';


export const ProjectForm = () => {
    const { project, isValid, isCameraSetupValid, isRobotSetupValid } = useProjectDataContext();

    const navigate = useNavigate();

    const saveMutation = $api.useMutation('put','/api/projects')

    const save = () => {
        console.log(project);
        saveMutation.mutateAsync({
            body: project
        }).then((projectId) => {
            navigate(paths.projects.edit({projectId}));
        })
    }

    return (
        <Flex width="100%" height="100%" direction={"column"} alignItems="center">
            <View flex={1} width={"100%"} maxWidth="1320px" paddingTop="size-400">
                <Flex justifyContent={"space-between"}>
                    <Heading>New Project</Heading>
                    <ButtonGroup>
                        <Button isDisabled={!isValid() || saveMutation.isPending} onPress={save}>Save</Button>
                    </ButtonGroup>
                </Flex>
                <Tabs aria-label="NewProject">
                    <TabList>
                        <Item key="Properties">
                            <Gear fill={'white'} />
                            <Text>Properties</Text>
                        </Item>
                        <Item key="Robots">
                            <Contract fill={isRobotSetupValid() ? 'white' : 'var(--spectrum-semantic-notice-color-icon)'} />
                            Robots
                        </Item>
                        <Item key="Cameras">
                            <FitScreen fill={isCameraSetupValid() ? 'white' : 'var(--spectrum-semantic-notice-color-icon)'} />
                            Cameras
                        </Item>
                    </TabList>
                    <TabPanels>
                        <Item key="Properties">
                            <PropertiesView />
                        </Item>
                        <Item key="Robots">
                            <RobotsView />
                        </Item>
                        <Item key="Cameras">
                            <CamerasView />
                        </Item>
                    </TabPanels>
                </Tabs>
            </View>
        </Flex>

    )
}

export const NewProject = () => {
    return (
        <ProjectDataContext.Provider value={createProjectDataContext()}>
            <ProjectForm />
        </ProjectDataContext.Provider>
    );
}