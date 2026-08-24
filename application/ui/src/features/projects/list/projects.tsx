import { useDeferredValue, useMemo, useState } from 'react';

import { Divider, Flex, Grid, View } from '@geti-ui/ui';
import { preload } from 'react-dom';

import { $api } from '../../../api/client';
import { SchemaProjectInput } from '../../../api/openapi-spec';
import backgroundUrl from '../../../assets/background.webp';
import { NoMatchingProjects } from './no-matching-projects';
import { NoProjects } from './no-projects/no-projects';
import { ProjectActions, type SortDirection } from './project-actions/project-actions';
import { ProjectCard } from './project-card';
import { ProjectsHeading } from './projects-heading/projects-heading';

import classes from './project-list.module.css';

// The background is applied through a CSS `background-image`, which the browser only starts
// fetching once an element matching `.container` is in the render tree. Because that element
// lives below a suspending query, the download would otherwise begin only after
// `GET /api/projects` resolves. Preloading at module scope emits a
// `<link rel="preload" as="image" fetchpriority="high">` as soon as the bundle evaluates, so the
// image is fetched in parallel with the request rather than after it.
preload(backgroundUrl, { as: 'image', fetchPriority: 'high' });

type ProjectsListProps = {
    projects: SchemaProjectInput[];
};

const ProjectsList = ({ projects }: ProjectsListProps) => {
    const [searchQuery, setSearchQuery] = useState<string>('');
    const deferredSearchQuery = useDeferredValue(searchQuery);
    const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

    const filteredProjects = useMemo(
        () =>
            projects.filter((project) => {
                return project.name.toLocaleLowerCase().includes(deferredSearchQuery.toLocaleLowerCase());
            }),
        [deferredSearchQuery, projects]
    );

    const sortedProjects = useMemo(() => {
        return filteredProjects.toSorted((projectA, projectB) => {
            const projectCreatedAtA = projectA.created_at ? new Date(projectA.created_at) : new Date(0);
            const projectUpdatedAtB = projectB.created_at ? new Date(projectB.created_at) : new Date(0);

            return sortDirection === 'desc'
                ? projectUpdatedAtB.getTime() - projectCreatedAtA.getTime()
                : projectCreatedAtA.getTime() - projectUpdatedAtB.getTime();
        });
    }, [filteredProjects, sortDirection]);

    return (
        <View height={'100%'} padding={'size-800'} paddingBottom={0}>
            <View position={'absolute'} top={0} left={0} right={0} bottom={0} UNSAFE_className={classes.container} />
            <Flex
                direction={'column'}
                height='100%'
                maxWidth={'240ch'}
                marginX='auto'
                position={'relative'}
                gap={'size-300'}
            >
                <Flex justifyContent={'center'} marginBottom={'size-400'}>
                    <ProjectsHeading />
                </Flex>

                <Divider size={'S'} />

                <ProjectActions
                    searchQuery={searchQuery}
                    onSearch={setSearchQuery}
                    projectsCount={sortedProjects.length}
                    totalProjectsCount={projects.length}
                    sortDirection={sortDirection}
                    onSortDirectionChange={setSortDirection}
                />

                {sortedProjects.length === 0 ? (
                    <NoMatchingProjects />
                ) : (
                    <Grid
                        flex={1}
                        minHeight={0}
                        width={'100%'}
                        gap={'size-300'}
                        marginX={'auto'}
                        columns={['1fr', '1fr']}
                        alignItems={'start'}
                        alignContent={'start'}
                        UNSAFE_className={classes.projectsListContainer}
                    >
                        {sortedProjects.map((item) => (
                            <ProjectCard key={item.id} item={item} isActive={false} />
                        ))}
                    </Grid>
                )}
            </Flex>
        </View>
    );
};

export const Projects = () => {
    const { data: projects } = $api.useSuspenseQuery('get', '/api/projects');

    if (projects.length === 0) {
        return <NoProjects />;
    }

    return <ProjectsList projects={projects} />;
};
