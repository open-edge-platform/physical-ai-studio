import { Grid, Heading, Text, View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { $api } from '../../../api/client';
import { NewProjectLink } from './new-project-link.component';
import { ProjectCard } from './project-card';

import classes from './/project-list.module.scss';

export const ProjectList = () => {
    const { data: projects } = $api.useSuspenseQuery('get', '/api/projects');

    return (
        <View padding='size-400' height='100%' maxWidth={'240ch'} marginX='auto'>
            <Heading
                level={1}
                marginBottom={'size-250'}
                UNSAFE_style={{
                    textAlign: 'center',
                    fontSize: 'var(--spectrum-global-dimension-font-size-700)',
                }}
            >
                Projects
            </Heading>

            <Text UNSAFE_className={classes.description}>
                To create a project, start by defining your objectives. Then, design the data flow to ensure proper
                processing at each stage. Implement the required tools and technologies for automation, and finally,
                test the project to confirm it runs smoothly and meets your goals.
            </Text>

            <Grid
                gap={'size-300'}
                marginX={'auto'}
                justifyContent={'center'}
                columns={isEmpty(projects) ? ['size-3600'] : ['1fr', '1fr']}
            >
                <NewProjectLink />

                {projects.map((item, index) => (
                    <ProjectCard key={item.id} item={item} isActive={index === 0} />
                ))}
            </Grid>
        </View>
    );
};
