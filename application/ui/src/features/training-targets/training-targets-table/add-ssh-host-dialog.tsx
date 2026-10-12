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
    Text,
    TextField,
} from '@geti-ui/ui';

import { getApiErrorMessage, getSshHostKeyFingerprint } from '../../../api/errors';
import { SchemaSshHostAliasOption } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { SshHostKeyConfirmationDialog } from './ssh-host-key-confirmation-dialog';
import { useAddSshHostMutation } from './use-add-ssh-host-mutation';

import classes from './add-ssh-host-dialog.module.css';

type AddSshHostDialogProps = {
    close: () => void;
    onCreated: (option: SchemaSshHostAliasOption) => void;
};

/**
 * Appends a new Host entry to the user's ~/.ssh/config and confirms Studio
 * can dial it, via `POST /api/remote-servers/aliases`.
 *
 * The host-key confirmation retry swaps this dialog's own content for
 * `SshHostKeyConfirmationDialog`; this dialog owns its `DialogTrigger`
 * lifecycle, so its parent does not need to handle the confirmation.
 */
export const AddSshHostDialog = ({ close, onCreated }: AddSshHostDialogProps) => {
    const [alias, setAlias] = useState('');
    const [hostname, setHostname] = useState('');
    const [port, setPort] = useState<number | undefined>(22);
    const [user, setUser] = useState('');
    const [identityFile, setIdentityFile] = useState('');
    const [hostKeyFingerprint, setHostKeyFingerprint] = useState<string | undefined>(undefined);
    const { save, reset, isPending, error } = useAddSshHostMutation();

    const submit = (acceptedHostKeyFingerprint?: string) => {
        save(
            {
                alias: alias.trim(),
                hostname: hostname.trim(),
                port: port ?? 22,
                user: user.trim() || null,
                identity_file: identityFile.trim() || null,
            },
            {
                onSuccess: (option) => {
                    onCreated(option);
                    close();
                },
                acceptedHostKeyFingerprint,
                onHostKeyConfirmationRequired: setHostKeyFingerprint,
            }
        );
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        submit();
    };

    if (hostKeyFingerprint !== undefined) {
        return (
            <SshHostKeyConfirmationDialog
                host={`${hostname.trim()}:${port ?? 22}`}
                fingerprint={hostKeyFingerprint}
                onConfirm={() => {
                    setHostKeyFingerprint(undefined);
                    submit(hostKeyFingerprint);
                }}
                onCancel={() => {
                    setHostKeyFingerprint(undefined);
                    reset();
                }}
            />
        );
    }

    const errorMessage =
        error && getSshHostKeyFingerprint(error) === undefined
            ? (getApiErrorMessage(error) ?? 'The SSH host could not be added. Try again.')
            : undefined;
    const canSubmit = alias.trim() !== '' && hostname.trim() !== '';

    return (
        <Form onSubmit={handleSubmit} validationBehavior='native'>
            <Dialog width='size-4600' UNSAFE_className={classes.dialog}>
                <Heading>Add SSH host</Heading>
                <Divider />
                <Content>
                    <Flex direction='column' gap='size-150'>
                        <Text>
                            Appends a new Host entry to your ~/.ssh/config, after confirming Studio can reach it.
                        </Text>
                        <TextField
                            // eslint-disable-next-line jsx-a11y/no-autofocus
                            autoFocus
                            isRequired
                            label='Alias'
                            value={alias}
                            onChange={setAlias}
                            description='Name for the new Host entry.'
                            width='100%'
                        />
                        <TextField isRequired label='Host' value={hostname} onChange={setHostname} width='100%' />
                        <NumberField
                            label='Port'
                            value={port}
                            onChange={setPort}
                            minValue={1}
                            maxValue={65535}
                            formatOptions={{ useGrouping: false }}
                            width='100%'
                        />
                        <TextField label='User' value={user} onChange={setUser} width='100%' />
                        <TextField
                            label='Key path'
                            value={identityFile}
                            onChange={setIdentityFile}
                            description='Path to a private key file on this Studio host.'
                            width='100%'
                        />
                        {errorMessage !== undefined && <InlineAlert variant='error'>{errorMessage}</InlineAlert>}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        Cancel
                    </Button>
                    <Button variant='accent' type='submit' isDisabled={!canSubmit} isPending={isPending}>
                        Add host
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
