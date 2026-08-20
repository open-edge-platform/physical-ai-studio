import { useState } from 'react';

import { Badge, Button, Flex, Heading, Link, Text, View } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { SchemaPluginExtensionResponse, SchemaPluginResponse, SchemaPluginRobotResponse } from '../../api/openapi-spec';
import { Table, TableColumn } from '../../components/table/table';
import { usePluginActions, usePluginsQuery } from './plugins.hooks';

import classes from './plugins.module.css';

const PLUGIN_COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: 'minmax(0, 2fr)', header: 'Plugin' },
    { width: 'max-content', header: 'Robots' },
    { width: 'auto', align: 'end' },
];

const ROLE_CLASS_NAMES = {
    follower: classes.roleFollower,
    leader: classes.roleLeader,
} as const;

const RoleBadge = ({ role }: { role: SchemaPluginRobotResponse['role'] }) => (
    <Badge variant='neutral' UNSAFE_className={clsx(classes.roleBadge, ROLE_CLASS_NAMES[role])}>
        {role}
    </Badge>
);

const PluginRobots = ({ robots }: { robots: SchemaPluginRobotResponse[] }) => {
    if (robots.length === 0) {
        return <Text>Robots are discovered after installation.</Text>;
    }
    return (
        <Flex gap='size-200' wrap>
            {robots.map((robot) => (
                <View key={robot.type} padding='size-50' UNSAFE_className={classes.robotChip}>
                    <Flex alignItems='center' gap='size-100'>
                        <RoleBadge role={robot.role} />
                        <Text UNSAFE_className={classes.robotName}>{robot.display_name}</Text>
                    </Flex>
                </View>
            ))}
        </Flex>
    );
};

const ExtensionRow = ({
    extension,
    isBusy,
    busyId,
    onInstall,
    onUninstall,
}: {
    extension: SchemaPluginExtensionResponse;
    isBusy: boolean;
    busyId: string | undefined;
    onInstall: (pluginId: string) => void;
    onUninstall: (pluginId: string) => void;
}) => {
    const isThisBusy = busyId === extension.id && isBusy;
    return (
        <View padding='size-100' UNSAFE_className={classes.extensionRow}>
            <Flex alignItems='center' justifyContent='space-between' gap='size-200'>
                <Flex direction='column' gap='size-50' flex={1}>
                    <Flex alignItems='center' gap='size-100'>
                        <Heading level={4} margin={0}>
                            {extension.name}
                        </Heading>
                        {extension.installed ? (
                            <Badge variant='positive'>v{extension.installed_version}</Badge>
                        ) : (
                            <Badge variant='neutral'>Available</Badge>
                        )}
                    </Flex>
                    <Text UNSAFE_className={classes.extensionDescription}>{extension.description}</Text>
                </Flex>
                {extension.installed ? (
                    <Button variant='secondary' isDisabled={isBusy} onPress={() => onUninstall(extension.id)}>
                        {isThisBusy ? 'Uninstalling…' : 'Uninstall'}
                    </Button>
                ) : (
                    <Button variant='secondary' isDisabled={isBusy} onPress={() => onInstall(extension.id)}>
                        {isThisBusy ? 'Installing…' : 'Install'}
                    </Button>
                )}
            </Flex>
        </View>
    );
};

const PluginDetail = ({
    plugin,
    isBusy,
    busyId,
    onInstall,
    onUninstall,
}: {
    plugin: SchemaPluginResponse;
    isBusy: boolean;
    busyId: string | undefined;
    onInstall: (pluginId: string) => void;
    onUninstall: (pluginId: string) => void;
}) => {
    const extensions = plugin.extensions ?? [];
    return (
        <View backgroundColor='gray-75' padding='size-300' borderColor='gray-300' borderWidth='thin'>
            <Flex direction='column' gap='size-200'>
                {plugin.installed && plugin.in_use_robot_count > 0 ? (
                    <Text UNSAFE_className={classes.inUse}>
                        In use by {plugin.in_use_robot_count} robot{plugin.in_use_robot_count === 1 ? '' : 's'}
                    </Text>
                ) : null}
                <Flex direction='column' gap='size-100'>
                    <Heading level={4}>Robots</Heading>
                    <PluginRobots robots={plugin.robots} />
                </Flex>
                {plugin.repo_url ? (
                    <Link href={plugin.repo_url} target='_blank' rel='noreferrer'>
                        GitHub
                    </Link>
                ) : null}
                {extensions.length > 0 ? (
                    plugin.installed ? (
                        <Flex direction='column' gap='size-100'>
                            <Heading level={4}>Extensions</Heading>
                            {extensions.map((extension) => (
                                <ExtensionRow
                                    key={extension.id}
                                    extension={extension}
                                    isBusy={isBusy}
                                    busyId={busyId}
                                    onInstall={onInstall}
                                    onUninstall={onUninstall}
                                />
                            ))}
                        </Flex>
                    ) : (
                        <Text UNSAFE_className={classes.extensionHint}>
                            {extensions.length} extension{extensions.length === 1 ? '' : 's'} become available after
                            installing this plugin.
                        </Text>
                    )
                ) : null}
            </Flex>
        </View>
    );
};

type PluginRowProps = {
    plugin: SchemaPluginResponse;
    isBusy: boolean;
    busyId: string | undefined;
    isExpanded: boolean;
    onExpandedChange: (isExpanded: boolean) => void;
    onInstall: (pluginId: string) => void;
    onUninstall: (pluginId: string) => void;
};

const PluginRow = ({
    plugin,
    isBusy,
    busyId,
    isExpanded,
    onExpandedChange,
    onInstall,
    onUninstall,
}: PluginRowProps) => {
    const isInstalled = plugin.installed;
    const isInUse = plugin.in_use_robot_count > 0;
    const isInstalling = busyId === plugin.id && isBusy;
    return (
        <Table.ExpandableRow
            id={`plugin-row-${plugin.id}`}
            label={plugin.name}
            isExpanded={isExpanded}
            onExpandedChange={onExpandedChange}
            detail={
                <PluginDetail
                    plugin={plugin}
                    isBusy={isBusy}
                    busyId={busyId}
                    onInstall={onInstall}
                    onUninstall={onUninstall}
                />
            }
        >
            <Flex direction='column' gap='size-50'>
                <Heading level={4} margin={0}>
                    {plugin.name}
                </Heading>
                <Text UNSAFE_className={classes.pluginDescription}>{plugin.description}</Text>
            </Flex>

            <Text UNSAFE_className={classes.robotCount}>
                {plugin.robots.length} robot{plugin.robots.length === 1 ? '' : 's'}
            </Text>

            <div onClick={(event) => event.stopPropagation()}>
                <Flex gap='size-100' alignItems='center' wrap>
                    {isInstalled ? (
                        <Button
                            variant='secondary'
                            isDisabled={isInUse || isBusy}
                            onPress={() => onUninstall(plugin.id)}
                        >
                            Uninstall
                        </Button>
                    ) : (
                        <Button variant='primary' isDisabled={isBusy} onPress={() => onInstall(plugin.id)}>
                            {isInstalling ? 'Installing…' : 'Install'}
                        </Button>
                    )}
                </Flex>
            </div>
        </Table.ExpandableRow>
    );
};

type PluginsTableProps = {
    plugins: SchemaPluginResponse[];
    isBusy: boolean;
    busyId: string | undefined;
    onInstall: (pluginId: string) => void;
    onUninstall: (pluginId: string) => void;
};

export const PluginsTable = ({ plugins, isBusy, busyId, onInstall, onUninstall }: PluginsTableProps) => {
    const [expandedPluginId, setExpandedPluginId] = useState<string | undefined>(undefined);

    return (
        <Table columns={PLUGIN_COLUMNS} isEmphasized>
            {plugins.map((plugin) => (
                <PluginRow
                    key={plugin.id}
                    plugin={plugin}
                    isBusy={isBusy}
                    busyId={busyId}
                    isExpanded={expandedPluginId === plugin.id}
                    onExpandedChange={(isExpanded) => setExpandedPluginId(isExpanded ? plugin.id : undefined)}
                    onInstall={onInstall}
                    onUninstall={onUninstall}
                />
            ))}
        </Table>
    );
};

export const PluginsView = () => {
    const pluginsQuery = usePluginsQuery();
    const { isBusy, busyId, install, uninstall } = usePluginActions();

    const plugins = pluginsQuery.data;

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Flex marginBottom={'size-250'} justifyContent={'space-between'} alignItems={'center'}>
                <View>
                    <Heading level={1}>Plugins</Heading>
                    <Text>Install and manage plugins for the server.</Text>
                </View>
            </Flex>
            <View UNSAFE_className={classes.container}>
                {plugins.length === 0 ? (
                    <Text UNSAFE_className={classes.emptyList}>No plugins are configured.</Text>
                ) : (
                    <PluginsTable
                        plugins={plugins}
                        isBusy={isBusy}
                        busyId={busyId}
                        onInstall={install}
                        onUninstall={uninstall}
                    />
                )}
            </View>
        </View>
    );
};
