import { Grid, Heading, Text, View } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';
import { preload } from 'react-dom';

import { $api } from '../../../api/client';
import backgroundUrl from '../../../assets/background.webp';
import { NewProjectLink } from './new-project-link.component';
import { NoProjects } from './no-projects/no-projects';
import { ProjectCard } from './project-card';

import classes from './project-list.module.css';

// The background is applied through a CSS `background-image`, which the browser only starts
// fetching once an element matching `.container` is in the render tree. Because that element
// lives below a suspending query, the download would otherwise begin only after
// `GET /api/projects` resolves. Preloading at module scope emits a
// `<link rel="preload" as="image" fetchpriority="high">` as soon as the bundle evaluates, so the
// image is fetched in parallel with the request rather than after it.
preload(backgroundUrl, { as: 'image', fetchPriority: 'high' });

export const ProjectList = () => {
    const { data: projects } = $api.useSuspenseQuery('get', '/api/projects');

    if (projects.length === 0) {
        return <NoProjects />;
    }

    return (
        <View height={'100%'}>
            <View position={'absolute'} top={0} left={0} right={0} bottom={0} UNSAFE_className={classes.container} />
            <View height='100%' maxWidth={'240ch'} marginX='auto' position={'relative'}>
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

                    {projects.map((item) => (
                        <ProjectCard key={item.id} item={item} isActive={false} />
                    ))}
                </Grid>
            </View>
        </View>
    );
};
