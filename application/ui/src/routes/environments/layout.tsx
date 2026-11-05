import { ActionButton, Flex, Grid, Heading, Item, Menu, MenuTrigger, toast, View } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { NavLink, Outlet, useParams } from 'react-router';

import { $api } from '../../api/client';
import { getApiErrorMessage, isResourceInUseError } from '../../api/errors';
import { SchemaEnvironmentOutput } from '../../api/openapi-spec';
import { AddResourceButton } from '../../components/add-resource-button/add-resource-button';
import { CollapsibleSidebarLayout } from '../../components/collapsible-sidebar-layout/collapsible-sidebar-layout';
import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../paths';

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
                        deleteEnvironmentMutation.mutate(
                            { params: { path: { project_id, environment_id } } },
                            {
                                onError: (error) => {
                                    const message = getApiErrorMessage(error) ?? 'Failed to delete environment.';
                                    if (isResourceInUseError(error)) {
                                        toast.info(message);
                                    } else {
                                        toast.negative(message);
                                    }
                                },
                            }
                        );
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
    );
};

export const Layout = () => {
    const { project_id } = useProjectId();

    return (
        <CollapsibleSidebarLayout
            toolbar={
                <AddResourceButton to={paths.project.environments.new({ project_id })}>
                    Configure new environment
                </AddResourceButton>
            }
            sidebar={<EnvironmentsList />}
        >
            <Outlet />
        </CollapsibleSidebarLayout>
    );
};
