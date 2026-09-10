import { FormEvent, ReactNode, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    ContextualHelp,
    Dialog,
    Divider,
    Flex,
    Form,
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

import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRemoteTrainer } from '../../../../api/openapi-spec';
import { useRemoteTrainerFormMutation } from './use-remote-trainer-form-mutation';
import { useSshHostAliases } from './use-ssh-host-aliases';

import classes from './remote-trainer-form.module.css';

type RemoteTrainerFormProps = {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
};

type SshHostSource = 'pick' | 'manual';

const InfoHelp = ({ title, children }: { title: string; children: ReactNode }) => (
    <ContextualHelp variant='info'>
        <Heading>{title}</Heading>
        <Content>
            <Text>{children}</Text>
        </Content>
    </ContextualHelp>
);

const parseUrlPort = (value: string): number | undefined => {
    try {
        const port = new URL(value).port;
        return port ? Number(port) : undefined;
    } catch {
        return undefined;
    }
};

export const RemoteTrainerForm = ({ remoteTrainer, close }: RemoteTrainerFormProps) => {
    const [name, setName] = useState(remoteTrainer?.name ?? '');
    const [url, setUrl] = useState(remoteTrainer?.url ?? '');
    const [sshTunnelEnabled, setSshTunnelEnabled] = useState(remoteTrainer?.ssh_host_alias !== undefined);
    const [sshHostSource, setSshHostSource] = useState<SshHostSource>('pick');
    const [sshHostAlias, setSshHostAlias] = useState(remoteTrainer?.ssh_host_alias ?? '');
    const [newAlias, setNewAlias] = useState('');
    const [newHostname, setNewHostname] = useState('');
    const [newPort, setNewPort] = useState<number | undefined>(22);
    const [newUser, setNewUser] = useState('');
    const [newIdentityFile, setNewIdentityFile] = useState('');
    const [sshRemotePort, setSshRemotePort] = useState<number | undefined>(remoteTrainer?.ssh_remote_port ?? undefined);
    const [sshLocalPort, setSshLocalPort] = useState<number | undefined>(remoteTrainer?.ssh_local_port ?? undefined);
    const isEditing = remoteTrainer !== undefined;
    const { aliases } = useSshHostAliases();
    const { save, isPending, error } = useRemoteTrainerFormMutation(remoteTrainer);

    const isManual = sshTunnelEnabled && sshHostSource === 'manual';

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        save(
            {
                name: name.trim(),
                url,
                ...(sshTunnelEnabled
                    ? {
                          ssh_host_alias: isManual ? newAlias.trim() : sshHostAlias.trim(),
                          ssh_remote_port: sshRemotePort,
                          ssh_local_port: sshLocalPort,
                      }
                    : { ssh_host_alias: undefined, ssh_remote_port: undefined, ssh_local_port: undefined }),
            },
            isManual
                ? {
                      alias: newAlias.trim(),
                      hostname: newHostname.trim(),
                      port: newPort ?? 22,
                      user: newUser.trim() || undefined,
                      identity_file: newIdentityFile.trim() || undefined,
                  }
                : undefined,
            { onSuccess: close }
        );
    };

    const errorMessage = error
        ? (getApiErrorMessage(error) ?? 'The remote trainer could not be saved. Try again.')
        : undefined;

    const hasValidSshHost = isManual ? newAlias.trim() !== '' && newHostname.trim() !== '' : sshHostAlias.trim() !== '';
    const canSubmit = name.trim() !== '' && url !== '' && (!sshTunnelEnabled || (hasValidSshHost && sshLocalPort));

    return (
        <Form onSubmit={handleSubmit} validationBehavior='native' width='size-6000'>
            <Dialog>
                <Heading>{isEditing ? 'Edit remote trainer' : 'Add remote trainer'}</Heading>
                <Divider />
                <Content>
                    <Flex direction='column' gap='size-200'>
                        <TextField
                            // eslint-disable-next-line jsx-a11y/no-autofocus
                            autoFocus
                            isRequired
                            label='Name'
                            value={name}
                            onChange={setName}
                            width='100%'
                        />
                        <TextField
                            isRequired
                            label='Trainer URL'
                            type='url'
                            value={url}
                            onChange={setUrl}
                            contextualHelp={
                                <InfoHelp title='Trainer URL'>
                                    Use the endpoint URL that accepts Physical AI Studio training jobs. When using an
                                    SSH tunnel, point this at the tunnel&apos;s local port, e.g. http://127.0.0.1:8001.
                                </InfoHelp>
                            }
                            width='100%'
                        />
                        <Switch
                            isSelected={sshTunnelEnabled}
                            onChange={(isSelected) => {
                                setSshTunnelEnabled(isSelected);
                                // Local port has no other sensible default, and the URL's port is the
                                // common case (see the hint text below) - prefill both from it so a new
                                // trainer does not need the same number typed in twice. Only fills a port
                                // that is not already set, so this never overwrites a saved trainer's
                                // values or something the user already typed.
                                const urlPort = isSelected ? parseUrlPort(url) : undefined;
                                if (urlPort !== undefined) {
                                    setSshRemotePort((current) => current ?? urlPort);
                                    setSshLocalPort((current) => current ?? urlPort);
                                }
                            }}
                        >
                            Reach this trainer through an SSH tunnel
                        </Switch>
                        {sshTunnelEnabled && (
                            <Flex direction='column' gap='size-100'>
                                <Text UNSAFE_className={classes.hint}>
                                    Studio never stores SSH credentials - only a <code>Host</code> name from{' '}
                                    <code>~/.ssh/config</code>.
                                </Text>
                                <RadioGroup
                                    label='SSH host'
                                    orientation='horizontal'
                                    isEmphasized
                                    value={sshHostSource}
                                    onChange={(value) => setSshHostSource(value as SshHostSource)}
                                >
                                    <Radio value='pick'>Pick from SSH config</Radio>
                                    <Radio value='manual'>Configure manually</Radio>
                                </RadioGroup>
                                {sshHostSource === 'pick' ? (
                                    <Picker
                                        isRequired
                                        label='SSH host alias'
                                        placeholder='Select...'
                                        selectedKey={sshHostAlias || null}
                                        onSelectionChange={(key) => setSshHostAlias(key ? String(key) : '')}
                                        contextualHelp={
                                            <InfoHelp title='SSH host alias'>
                                                Pick a Host entry from your ~/.ssh/config.
                                            </InfoHelp>
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
                                            value={newAlias}
                                            onChange={setNewAlias}
                                            contextualHelp={
                                                <InfoHelp title='SSH host alias'>
                                                    Name for the new SSH config Host entry.
                                                </InfoHelp>
                                            }
                                            width='100%'
                                        />
                                        <Flex gap='size-200'>
                                            <TextField
                                                isRequired
                                                label='Host'
                                                value={newHostname}
                                                onChange={setNewHostname}
                                                contextualHelp={
                                                    <InfoHelp title='Host'>
                                                        Hostname or IP address to connect to.
                                                    </InfoHelp>
                                                }
                                                width='100%'
                                            />
                                            <NumberField
                                                label='Port'
                                                value={newPort}
                                                onChange={setNewPort}
                                                minValue={1}
                                                maxValue={65535}
                                                width='100%'
                                            />
                                        </Flex>
                                        <Flex gap='size-200'>
                                            <TextField
                                                label='User'
                                                value={newUser}
                                                onChange={setNewUser}
                                                width='100%'
                                            />
                                            <TextField
                                                label='Key path'
                                                value={newIdentityFile}
                                                onChange={setNewIdentityFile}
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
                                        value={sshRemotePort}
                                        onChange={setSshRemotePort}
                                        minValue={1}
                                        maxValue={65535}
                                        contextualHelp={
                                            <InfoHelp title='Remote port'>Defaults to the URL&apos;s port.</InfoHelp>
                                        }
                                        width='100%'
                                    />
                                    <NumberField
                                        isRequired
                                        label='Local port'
                                        value={sshLocalPort}
                                        onChange={setSshLocalPort}
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
                        {errorMessage !== undefined && (
                            <Text UNSAFE_className={classes.errorMessage}>{errorMessage}</Text>
                        )}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        Cancel
                    </Button>
                    <Button variant='accent' type='submit' isDisabled={!canSubmit} isPending={isPending}>
                        {isEditing ? 'Save changes' : 'Add trainer'}
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
