import { Flex, Heading, Link, Text, View } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';

import { $api } from '../../api/client';
import { SchemaProjectConfig } from '../../api/openapi-spec';
import { paths } from '../../router';

import classes from './index.module.scss';

const NewProject = () => {
    return (
        <Link href={paths.projects.new({})} UNSAFE_className={classes.link}>
            <View
                borderColor={'gray-700'}
                borderWidth={'thin'}
                borderRadius={'regular'}
                backgroundColor={'gray-300'}
                width='100%'
                height='156px'
                UNSAFE_className={classes.dashed}
            >
                <Flex direction={'column'} justifyContent={'center'} alignItems={'center'} height='100%' gap='size-100'>
                    <AddCircle />
                    <Text>Add Project</Text>
                </Flex>
            </View>
        </Link>
    );
};

interface ProjectItemProps {
    project: SchemaProjectConfig;
}

const ProjectItem = ({ project }: ProjectItemProps) => {
    return (
        <Link href={paths.project.datasets.index({ project_id: project.id })} UNSAFE_className={classes.link}>
            <View
                borderColor={'gray-200'}
                borderWidth={'thin'}
                borderRadius={'regular'}
                width='100%'
                backgroundColor={'gray-50'}
                height='156px'
                padding='size-300'
            >
                <Flex direction={'column'} height={'100%'} justifyContent={'space-between'}>
                    <Heading>{project.name}</Heading>
                    <Flex direction={'column'}>
                        <Text>Datasets: {project.datasets.join(', ')}</Text>
                        <Text>Cameras: {project.cameras.map((c) => c.name).join(', ')}</Text>
                        <Text>Robots: {project.robots.map((c) => c.id).join(', ')}</Text>
                    </Flex>
                </Flex>
            </View>
        </Link>
    );
};

export const Index = () => {
    const { data: projects } = $api.useSuspenseQuery('get', '/api/projects');
    return (
        <Flex width='100%' height='100%' direction={'column'} alignItems='center'>
            <Flex direction={'column'} maxWidth='1320px' flex={0}>
                <View padding='size-400'>
                    <Flex direction={'column'} alignItems={'center'}>
                        <Heading>Projects</Heading>
                        <Text UNSAFE_style={{ textAlign: 'center' }}>
                            {' '}
                            Fusce vel imperdiet tellus. Nullam pulvinar sodales mauris. Sed vel euismod libero. Duis
                            condimentum, dolor finibus volutpat euismod, lorem nibh auctor sapien, eu vulputate dui nisi
                            ac dui. Cras gravida id erat non vulputate. Phasellus at pulvinar justo, vitae ornare justo.
                            Curabitur id ultrices dui, volutpat venenatis magna. Aliquam facilisis, erat nec vehicula
                            tincidunt, ipsum ipsum bibendum ex, at sagittis mi magna et lacus. Nunc in lacinia metus.
                            Aenean sodales lectus at massa finibus vestibulum. Curabitur quis eros interdum, lacinia
                            ante eget, molestie sem.{' '}
                        </Text>
                    </Flex>
                </View>
                <Flex direction={'row'} wrap={'wrap'} gap={'size-275'}>
                    <NewProject />
                    {projects.map((project) => (
                        <ProjectItem key={project.id} project={project} />
                    ))}
                </Flex>
            </Flex>
        </Flex>
    );
};
