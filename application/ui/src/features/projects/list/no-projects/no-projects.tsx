import { Button, Flex, Heading, Text, View } from '@geti-ui/ui';

import { ProjectsHeading } from '../projects-heading/projects-heading';
import { CreateProject } from './create-project';

import classes from './no-projects.module.css';

export const NoProjects = () => {
    return (
        <Flex justifyContent={'center'}>
            <View
                position={'absolute'}
                top={0}
                left={0}
                right={0}
                bottom={0}
                UNSAFE_className={classes.backgroundImg}
            />

            <Flex
                direction={'column'}
                alignItems={'center'}
                gap={'size-800'}
                position={'relative'}
                marginY={'size-1000'}
            >
                <ProjectsHeading />
                <View
                    width={'clamp(700px, 40vw, 1000px)'}
                    padding={'size-800'}
                    borderRadius={'regular'}
                    borderWidth={'thick'}
                    borderColor={'gray-400'}
                    UNSAFE_className={classes.createProjectCard}
                >
                    <Flex direction={'column'} alignItems={'center'} justifyContent={'center'} gap={'size-200'}>
                        <Heading UNSAFE_className={classes.createProjectTitle}>Create your first project</Heading>
                        <Text UNSAFE_className={classes.createProjectText}>
                            To create a project, start by defining your objectives. Then, design the data flow to ensure
                            proper processing at each stage. Implement the required tools and technologies for
                            automation, and finally, test the project to confirm it runs smoothly and meets your goals.
                        </Text>
                        <CreateProject trigger={<Button variant={'accent'}>Create new project</Button>} />
                    </Flex>
                </View>
            </Flex>
        </Flex>
    );
};
