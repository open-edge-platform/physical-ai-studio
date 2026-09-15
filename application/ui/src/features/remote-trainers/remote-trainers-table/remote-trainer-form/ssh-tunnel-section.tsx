import { Dispatch, ReactNode, SetStateAction } from 'react';

import {
    Content,
    ContextualHelp,
    Flex,
    Heading,
    Item,
    NumberField,
    Picker,
    Radio,
    RadioGroup,
    Switch,
    Text,
    TextField,
} from '@geti-ui/ui';

import { SchemaSshHostAliasOption } from '../../../../api/openapi-spec';

import classes from './remote-trainer-form.module.css';

export type SshHostSource = 'pick' | 'manual';

export type SshConfig = {
    tunnelEnabled: boolean;
    hostSource: SshHostSource;
    hostAlias: string;
    newAlias: string;
    newHostname: string;
    newPort: number | undefined;
    newUser: string;
    newIdentityFile: string;
    remotePort: number | undefined;
    localPort: number | undefined;
};

type RemoteTrainerSshFields = {
    ssh_host_alias?: string | null;
    ssh_remote_port?: number | null;
    ssh_local_port?: number | null;
};

export const initialSshConfig = (remoteTrainer?: RemoteTrainerSshFields): SshConfig => ({
    tunnelEnabled: Boolean(remoteTrainer?.ssh_host_alias),
    hostSource: 'pick',
    hostAlias: remoteTrainer?.ssh_host_alias ?? '',
    newAlias: '',
    newHostname: '',
    newPort: 22,
    newUser: '',
    newIdentityFile: '',
    remotePort: remoteTrainer?.ssh_remote_port ?? undefined,
    localPort: remoteTrainer?.ssh_local_port ?? undefined,
});

export const parseUrlPort = (value: string): number | undefined => {
    try {
        const port = new URL(value).port;
        return port ? Number(port) : undefined;
    } catch {
        return undefined;
    }
};

export const InfoHelp = ({ title, children }: { title: string; children: ReactNode }) => (
    <ContextualHelp variant='info'>
        <Heading>{title}</Heading>
        <Content>
            <Text>{children}</Text>
        </Content>
    </ContextualHelp>
);

type SshTunnelSectionProps = {
    url: string;
    sshConfig: SshConfig;
    setSshConfig: Dispatch<SetStateAction<SshConfig>>;
    aliases: SchemaSshHostAliasOption[];
};

export const SshTunnelSection = ({ url, sshConfig, setSshConfig, aliases }: SshTunnelSectionProps) => {
    const update = (patch: Partial<SshConfig>) => setSshConfig((current) => ({ ...current, ...patch }));

    return (
        <>
            <Switch
                isSelected={sshConfig.tunnelEnabled}
                onChange={(isSelected) => {
                    // Local port has no other sensible default, and the URL's port is the
                    // common case (see the hint text below) - prefill both from it so a new
                    // trainer does not need the same number typed in twice. Only fills a port
                    // that is not already set, so this never overwrites a saved trainer's
                    // values or something the user already typed.
                    const urlPort = isSelected ? parseUrlPort(url) : undefined;
                    setSshConfig((current) => ({
                        ...current,
                        tunnelEnabled: isSelected,
                        remotePort: urlPort !== undefined ? (current.remotePort ?? urlPort) : current.remotePort,
                        localPort: urlPort !== undefined ? (current.localPort ?? urlPort) : current.localPort,
                    }));
                }}
            >
                Reach this trainer through an SSH tunnel
            </Switch>
            {sshConfig.tunnelEnabled && (
                <Flex direction='column' gap='size-100'>
                    <Text UNSAFE_className={classes.hint}>
                        Studio never stores SSH credentials - only a <code>Host</code> name from{' '}
                        <code>~/.ssh/config</code>.
                    </Text>
                    <RadioGroup
                        label='SSH host'
                        orientation='horizontal'
                        isEmphasized
                        value={sshConfig.hostSource}
                        onChange={(value) => update({ hostSource: value as SshHostSource })}
                    >
                        <Radio value='pick'>Pick from SSH config</Radio>
                        <Radio value='manual'>Configure manually</Radio>
                    </RadioGroup>
                    {sshConfig.hostSource === 'pick' ? (
                        <Picker
                            isRequired
                            label='SSH host alias'
                            placeholder='Select...'
                            selectedKey={sshConfig.hostAlias || null}
                            onSelectionChange={(key) => update({ hostAlias: key ? String(key) : '' })}
                            contextualHelp={
                                <InfoHelp title='SSH host alias'>Pick a Host entry from your ~/.ssh/config.</InfoHelp>
                            }
                            width='100%'
                        >
                            {aliases.map((option) => (
                                <Item key={option.alias} textValue={option.alias}>
                                    {option.hostname && option.hostname !== option.alias
                                        ? `${option.alias} (${option.hostname})`
                                        : option.alias}
                                </Item>
                            ))}
                        </Picker>
                    ) : (
                        <>
                            <TextField
                                isRequired
                                label='SSH host alias'
                                value={sshConfig.newAlias}
                                onChange={(value) => update({ newAlias: value })}
                                contextualHelp={
                                    <InfoHelp title='SSH host alias'>Name for the new SSH config Host entry.</InfoHelp>
                                }
                                width='100%'
                            />
                            <Flex gap='size-200'>
                                <TextField
                                    isRequired
                                    label='Host'
                                    value={sshConfig.newHostname}
                                    onChange={(value) => update({ newHostname: value })}
                                    contextualHelp={
                                        <InfoHelp title='Host'>Hostname or IP address to connect to.</InfoHelp>
                                    }
                                    width='100%'
                                />
                                <NumberField
                                    label='Port'
                                    value={sshConfig.newPort}
                                    onChange={(value) => update({ newPort: value })}
                                    minValue={1}
                                    maxValue={65535}
                                    width='100%'
                                />
                            </Flex>
                            <Flex gap='size-200'>
                                <TextField
                                    label='User'
                                    value={sshConfig.newUser}
                                    onChange={(value) => update({ newUser: value })}
                                    width='100%'
                                />
                                <TextField
                                    label='Key path'
                                    value={sshConfig.newIdentityFile}
                                    onChange={(value) => update({ newIdentityFile: value })}
                                    contextualHelp={
                                        <InfoHelp title='Key path'>
                                            Path to a private key file on this Studio host.
                                        </InfoHelp>
                                    }
                                    width='100%'
                                />
                            </Flex>
                        </>
                    )}
                    <Flex gap='size-200'>
                        <NumberField
                            label='Remote port'
                            value={sshConfig.remotePort}
                            onChange={(value) => update({ remotePort: value })}
                            minValue={1}
                            maxValue={65535}
                            contextualHelp={<InfoHelp title='Remote port'>Defaults to the URL&apos;s port.</InfoHelp>}
                            width='100%'
                        />
                        <NumberField
                            isRequired
                            label='Local port'
                            value={sshConfig.localPort}
                            onChange={(value) => update({ localPort: value })}
                            minValue={1}
                            maxValue={65535}
                            contextualHelp={
                                <InfoHelp title='Local port'>
                                    Loopback port the tunnel binds to on this Studio host.
                                </InfoHelp>
                            }
                            width='100%'
                        />
                    </Flex>
                </Flex>
            )}
        </>
    );
};
