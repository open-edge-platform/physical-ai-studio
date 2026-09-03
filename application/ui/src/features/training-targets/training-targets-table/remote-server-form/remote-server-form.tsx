import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Form, Heading, Text, TextField } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRemoteServer } from '../../../../api/openapi-spec';
import { InlineAlert } from '../../../robots/setup-wizard/shared/inline-alert';
import { isSshDeviceType, SshDeviceType, SshTargetFields } from './ssh-target-fields';
import { useRemoteServerFormMutation } from './use-remote-server-form-mutation';
import { VerifyAfterSaveDialog } from './verify-after-save-dialog';

import classes from './remote-server-form.module.css';

type RemoteServerFormProps = {
    remoteServer?: SchemaRemoteServer;
    close: () => void;
};

export const RemoteServerForm = ({ remoteServer, close }: RemoteServerFormProps) => {
    const [name, setName] = useState(remoteServer?.name ?? '');
    const [sshHostAlias, setSshHostAlias] = useState<string | undefined>(remoteServer?.ssh_host_alias);
    const [deviceType, setDeviceType] = useState<SshDeviceType | undefined>(
        remoteServer && isSshDeviceType(remoteServer.device_type) ? remoteServer.device_type : undefined
    );
    const isEditing = remoteServer !== undefined;
    const { save, isPending, error } = useRemoteServerFormMutation(remoteServer);
    const [savedServer, setSavedServer] = useState<SchemaRemoteServer | undefined>(undefined);

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (sshHostAlias === undefined || deviceType === undefined) {
            return;
        }

        save(
            { name: name.trim(), ssh_host_alias: sshHostAlias, device_type: deviceType },
            {
                onSuccess: (saved) => {
                    // Only a brand-new server needs the prompt: an edit changes
                    // connection details on a server that may already be verified,
                    // and re-prompting every time it's edited would be noise.
                    if (isEditing) {
                        close();
                        return;
                    }
                    setSavedServer(saved);
                },
            }
        );
    };

    if (savedServer !== undefined) {
        return <VerifyAfterSaveDialog savedServer={savedServer} close={close} />;
    }

    const errorMessage = error
        ? (getApiErrorMessage(error) ?? 'The training target could not be saved. Try again.')
        : undefined;

    return (
        <Form onSubmit={handleSubmit} validationBehavior='native' width='size-6000'>
            <Dialog>
                <Heading>{isEditing ? 'Edit SSH server' : 'Add SSH server'}</Heading>
                <Divider />
                <Content>
                    <Flex direction='column' gap='size-200'>
                        <Text UNSAFE_className={classes.hint}>
                            Studio never receives your SSH credentials. Pick a Host entry you already have in
                            ~/.ssh/config; keys stay on disk (or in your SSH agent).
                        </Text>
                        <InlineAlert variant='warning'>
                            <strong>Security risk:</strong> SSH servers have no built-in authentication. Anyone who can
                            reach this Studio backend can run arbitrary code as root on the server. Only connect to
                            servers you trust, and only run Studio on a single-user, localhost-only workstation.
                        </InlineAlert>
                        <TextField
                            // eslint-disable-next-line jsx-a11y/no-autofocus
                            autoFocus
                            isRequired
                            label='Name'
                            value={name}
                            onChange={setName}
                            width='100%'
                        />
                        <SshTargetFields
                            sshHostAlias={sshHostAlias}
                            onSshHostAliasChange={setSshHostAlias}
                            deviceType={deviceType}
                            onDeviceTypeChange={setDeviceType}
                        />
                        {errorMessage !== undefined && <InlineAlert variant='error'>{errorMessage}</InlineAlert>}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        Cancel
                    </Button>
                    <Button
                        variant='accent'
                        type='submit'
                        isDisabled={!name.trim() || sshHostAlias === undefined || deviceType === undefined}
                        isPending={isPending}
                    >
                        {isEditing ? 'Save changes' : 'Verify & save'}
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
