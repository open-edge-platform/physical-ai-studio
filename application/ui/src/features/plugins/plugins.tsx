import { Button, Flex, Heading, Text, View } from '@geti-ui/ui';

import { SchemaPluginResponse } from '../../api/openapi-spec';
import { Table, TableColumn } from '../../components/table/table';
import { usePluginActions, usePluginsQuery } from './plugins.hooks';

import classes from './plugins.module.css';

const PLUGIN_COLUMNS: TableColumn[] = [
    { width: 'minmax(0, 2fr)', header: 'Plugin' },
    { width: 'auto', align: 'end' },
];

type PluginRowProps = {
    plugin: SchemaPluginResponse;
    isBusy: boolean;
    busyId: string | undefined;
    onInstall: (pluginId: string) => void;
    onUninstall: (pluginId: string) => void;
};

const PluginRow = ({ plugin, isBusy, busyId, onInstall, onUninstall }: PluginRowProps) => {
    const isInstalled = plugin.installed;
    const isInstalling = busyId === plugin.id && isBusy;
    return (
        <Table.Row id={`plugin-row-${plugin.id}`}>
            <Flex direction='column' gap='size-50'>
                <Heading level={4} margin={0}>
                    {plugin.name}
                </Heading>
                <Text UNSAFE_className={classes.pluginDescription}>{plugin.description}</Text>
            </Flex>

            <div onClick={(event) => event.stopPropagation()}>
                <Flex gap='size-100' alignItems='center' wrap>
                    {isInstalled ? (
                        <Button variant='secondary' isDisabled={isBusy} onPress={() => onUninstall(plugin.id)}>
                            Uninstall
                        </Button>
                    ) : (
                        <Button variant='primary' isDisabled={isBusy} onPress={() => onInstall(plugin.id)}>
                            {isInstalling ? 'Installing…' : 'Install'}
                        </Button>
                    )}
                </Flex>
            </div>
        </Table.Row>
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
    return (
        <Table columns={PLUGIN_COLUMNS} isEmphasized>
            {plugins.map((plugin) => (
                <PluginRow
                    key={plugin.id}
                    plugin={plugin}
                    isBusy={isBusy}
                    busyId={busyId}
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
