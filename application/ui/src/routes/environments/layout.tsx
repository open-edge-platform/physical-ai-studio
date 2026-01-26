import { Suspense } from 'react';

import {
    ActionButton,
    Button,
    Divider,
    Flex,
    Grid,
    Heading,
    Icon,
    Item,
    Loading,
    Menu,
    MenuTrigger,
    minmax,
    View,
} from '@geti/ui';
import { Add, MoreMenu } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { NavLink, Outlet, useParams } from 'react-router-dom';

import { $api } from '../../api/client';
import { SchemaEnvironmentOutput } from '../../api/openapi-spec';
import { useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';

import classes from './../../features/robots/robots-list.module.scss';

const MenuActions = ({ environment_id }: { environment_id: string }) => {
    const { project_id } = useProjectId();
    const deleteEnvironmentMutation = $api.useMutation(
        'delete',
        '/api/projects/{project_id}/environments/{environment_id}'
    );

    return (
        <MenuTrigger>
            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
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
        <Flex direction='column' gap='size-100'>
            {/* TODO:  */}
            <View isHidden>
                <Flex justifyContent={'space-between'} alignItems={'end'}>
                    <span>Step 3: create an environment</span>
                    <Button>Next</Button>
                </Flex>
                <Divider size='S' marginY='size-200' />
            </View>

            <Button
                variant='secondary'
                href={paths.project.environments.new({ project_id })}
                UNSAFE_className={classes.addNewRobotButton}
            >
                <Icon marginEnd='size-50'>
                    <Add />
                </Icon>
                Configure a new environment
            </Button>

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
        <Grid areas={['camera controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='camera' backgroundColor={'gray-100'} padding='size-400'>
                <EnvironmentsList />
            </View>
            <View gridArea='controls' backgroundColor={'gray-50'}>
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
