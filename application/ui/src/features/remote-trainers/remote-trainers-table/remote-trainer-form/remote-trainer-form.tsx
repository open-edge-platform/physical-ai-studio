import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Form, Heading, Text, TextField } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRemoteTrainer } from '../../../../api/openapi-spec';
import { InfoHelp, initialSshConfig, SshTunnelSection } from './ssh-tunnel-section';
import { useRemoteTrainerFormMutation } from './use-remote-trainer-form-mutation';
import { useSshHostAliases } from './use-ssh-host-aliases';

import classes from './remote-trainer-form.module.css';

type RemoteTrainerFormProps = {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
};

export const RemoteTrainerForm = ({ remoteTrainer, close }: RemoteTrainerFormProps) => {
    const [name, setName] = useState(remoteTrainer?.name ?? '');
    const [url, setUrl] = useState(remoteTrainer?.url ?? '');
    const [sshConfig, setSshConfig] = useState(() => initialSshConfig(remoteTrainer));
    const isEditing = remoteTrainer !== undefined;
    const { aliases } = useSshHostAliases();
    const { save, isPending, error } = useRemoteTrainerFormMutation(remoteTrainer);

    const isManual = sshConfig.tunnelEnabled && sshConfig.hostSource === 'manual';

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        save(
            {
                name: name.trim(),
                url,
                ...(sshConfig.tunnelEnabled
                    ? {
                          ssh_host_alias: isManual ? sshConfig.newAlias.trim() : sshConfig.hostAlias.trim(),
                          ssh_remote_port: sshConfig.remotePort,
                          ssh_local_port: sshConfig.localPort,
                      }
                    : { ssh_host_alias: undefined, ssh_remote_port: undefined, ssh_local_port: undefined }),
            },
            isManual
                ? {
                      alias: sshConfig.newAlias.trim(),
                      hostname: sshConfig.newHostname.trim(),
                      port: sshConfig.newPort ?? 22,
                      user: sshConfig.newUser.trim() || undefined,
                      identity_file: sshConfig.newIdentityFile.trim() || undefined,
                  }
                : undefined,
            { onSuccess: close }
        );
    };

    const errorMessage = error
        ? (getApiErrorMessage(error) ?? 'The remote trainer could not be saved. Try again.')
        : undefined;

    const hasValidSshHost = isManual
        ? sshConfig.newAlias.trim() !== '' && sshConfig.newHostname.trim() !== ''
        : sshConfig.hostAlias.trim() !== '';
    const canSubmit =
        name.trim() !== '' && url !== '' && (!sshConfig.tunnelEnabled || (hasValidSshHost && sshConfig.localPort));

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
                        <SshTunnelSection
                            url={url}
                            sshConfig={sshConfig}
                            setSshConfig={setSshConfig}
                            aliases={aliases}
                        />
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
