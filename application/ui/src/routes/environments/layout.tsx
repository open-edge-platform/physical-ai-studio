import { Suspense } from 'react';

import { ActionButton, Flex, Grid, Heading, Item, Loading, Menu, MenuTrigger, minmax, Text, View } from '@geti-ui/ui';
import { Add, MoreMenu } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { Link, NavLink, Outlet, useParams } from 'react-router';

import { $api } from '../../api/client';
import { SchemaEnvironmentOutput } from '../../api/openapi-spec';
import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';

import classes from './../../features/robots/robots-list.module.css';

const MenuActions = ({ environment_id }: { environment_id: string }) => {
    const { project_id } = useProjectId();
    const deleteEnvironmentMutation = $api.useMutation(
        'delete',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            meta: {
                invalidates: [['get', '/api/projects/{project_id}/environments', { params: { path: { project_id } } }]],
            },
        }
    );

    return (
        <MenuTrigger>
            <ActionButton isQuiet>
                <MoreMenu />
            </ActionButton>
            <Menu
                selectionMode='single'
                onAction={(action) => {
                    if (action === 'delete') {
                        deleteEnvironmentMutation.mutate({ params: { path: { project_id, environment_id } } });
                    }
                }}
            >
                <Item href={paths.project.environments.edit({ project_id, environment_id })}>Edit</Item>
                <Item key='delete'>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};

const EnvironmentListItem = ({
    environment,
    isActive,
}: {
    environment: SchemaEnvironmentOutput;
    isActive: boolean;
}) => {
    return (
        <View
            padding='size-200'
            UNSAFE_className={clsx({
                [classes.robotListItem]: true,
                [classes.robotListItemActive]: isActive,
            })}
        >
            <Flex direction={'column'} justifyContent={'space-between'} gap={'size-50'}>
                <Grid areas={['name menu']} columns={['auto', '1fr']} gap={'size-100'}>
                    <View gridArea='name'>
                        <Heading level={2} UNSAFE_style={isActive ? { color: 'var(--energy-blue)' } : {}}>
                            {environment.name}
                        </Heading>
                    </View>
                    <View gridArea='menu' alignSelf={'end'} justifySelf={'end'}>
                        <MenuActions environment_id={environment.id} />
                    </View>
                </Grid>
            </Flex>
        </View>
    );
};

export const EnvironmentsList = () => {
    const { project_id = '' } = useParams<{ project_id: string }>();
    const {} = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    const environmentsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });

    return (
        <Flex direction='column' gap='size-200'>
            <Link to={paths.project.environments.new({ project_id })} className={classes.addNewRobotButton}>
                <Flex alignItems={'center'} justifyContent={'center'} gap={'size-75'}>
                    <Add />
                    <Text UNSAFE_style={{ lineHeight: '1.2' }}>Configure a new environment</Text>
                </Flex>
            </Link>

            <Flex direction='column' gap='size-100'>
                {environmentsQuery.data.map((environment) => {
                    const to = paths.project.environments.show({ project_id, environment_id: environment.id });

                    return (
                        <NavLink key={environment.id} to={to}>
                            {({ isActive }) => {
                                return <EnvironmentListItem environment={environment} isActive={isActive} />;
                            }}
                        </NavLink>
                    );
                })}
            </Flex>
        </Flex>
    );
};

export const Layout = () => {
    return (
        <Grid
            areas={['environments controls']}
            columns={['size-6000', minmax(0, '1fr')]}
            rows={[minmax(0, '1fr')]}
            height={'100%'}
            minHeight={0}
        >
            <View gridArea='environments' backgroundColor={'gray-100'} padding='size-400'>
                <EnvironmentsList />
            </View>
            <View gridArea='controls' backgroundColor={'gray-50'} minHeight={0} minWidth={0} overflow='auto'>
                <Suspense
                    fallback={
                        <Grid width='100%' height='100%'>
                            <Loading mode='inline' />
                        </Grid>
                    }
                >
                    <Outlet />
                </Suspense>
            </View>
        </Grid>
    );
};
