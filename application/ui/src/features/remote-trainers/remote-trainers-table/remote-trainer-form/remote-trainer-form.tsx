import { FormEvent, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Form,
    Heading,
    NumberField,
    Switch,
    Text,
    TextField,
} from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRemoteTrainer } from '../../../../api/openapi-spec';
import { useRemoteTrainerFormMutation } from './use-remote-trainer-form-mutation';

import classes from './remote-trainer-form.module.css';

type RemoteTrainerFormProps = {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
};

export const RemoteTrainerForm = ({ remoteTrainer, close }: RemoteTrainerFormProps) => {
    const [name, setName] = useState(remoteTrainer?.name ?? '');
    const [url, setUrl] = useState(remoteTrainer?.url ?? '');
    const [sshTunnelEnabled, setSshTunnelEnabled] = useState(remoteTrainer?.ssh_host_alias !== undefined);
    const [sshHostAlias, setSshHostAlias] = useState(remoteTrainer?.ssh_host_alias ?? '');
    const [sshRemotePort, setSshRemotePort] = useState<number | undefined>(remoteTrainer?.ssh_remote_port ?? undefined);
    const [sshLocalPort, setSshLocalPort] = useState<number | undefined>(remoteTrainer?.ssh_local_port ?? undefined);
    const isEditing = remoteTrainer !== undefined;
    const { save, isPending, error } = useRemoteTrainerFormMutation(remoteTrainer);

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        save(
            {
                name: name.trim(),
                url,
                ...(sshTunnelEnabled
                    ? {
                          ssh_host_alias: sshHostAlias.trim(),
                          ssh_remote_port: sshRemotePort,
                          ssh_local_port: sshLocalPort,
                      }
                    : { ssh_host_alias: undefined, ssh_remote_port: undefined, ssh_local_port: undefined }),
            },
            { onSuccess: close }
        );
    };

    const errorMessage = error
        ? (getApiErrorMessage(error) ?? 'The remote trainer could not be saved. Try again.')
        : undefined;

    const canSubmit =
        name.trim() !== '' && url !== '' && (!sshTunnelEnabled || (sshHostAlias.trim() !== '' && sshLocalPort));

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
                            description='Use the endpoint URL that accepts Physical AI Studio training jobs.'
                            width='100%'
                        />
                        <Switch isSelected={sshTunnelEnabled} onChange={setSshTunnelEnabled}>
                            Reach this trainer through an SSH tunnel
                        </Switch>
                        {sshTunnelEnabled && (
                            <Flex direction='column' gap='size-100'>
                                <Text UNSAFE_className={classes.hint}>
                                    Studio never stores an SSH key, password, or passphrase - only the name of a{' '}
                                    <code>Host</code> entry in your own <code>~/.ssh/config</code>. Studio keeps a
                                    standing tunnel open; point the URL above at its local port, e.g.{' '}
                                    <code>http://127.0.0.1:8001</code>.
                                </Text>
                                <TextField
                                    isRequired
                                    label='SSH host alias'
                                    value={sshHostAlias}
                                    onChange={setSshHostAlias}
                                    description='Name of a Host entry in your SSH config.'
                                    width='100%'
                                />
                                <Flex gap='size-200'>
                                    <NumberField
                                        label='Remote port'
                                        value={sshRemotePort}
                                        onChange={setSshRemotePort}
                                        minValue={1}
                                        maxValue={65535}
                                        description="Defaults to the URL's port."
                                        width='100%'
                                    />
                                    <NumberField
                                        isRequired
                                        label='Local port'
                                        value={sshLocalPort}
                                        onChange={setSshLocalPort}
                                        minValue={1}
                                        maxValue={65535}
                                        description='Loopback port the tunnel binds to on this Studio host.'
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
