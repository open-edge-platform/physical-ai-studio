import { FormEvent, useMemo, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Form,
    Heading,
    Item,
    Picker,
    Text,
    TextField,
} from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRemoteServer } from '../../../../api/openapi-spec';
import { InlineAlert } from '../../../robots/setup-wizard/shared/inline-alert';
import { useRemoteServerFormMutation } from './use-remote-server-form-mutation';

import classes from './remote-server-form.module.css';

const DEVICE_TYPES = ['cuda', 'xpu'] as const;

const isSshDeviceType = (value: string): value is (typeof DEVICE_TYPES)[number] =>
    (DEVICE_TYPES as readonly string[]).includes(value);

type RemoteServerFormProps = {
    remoteServer?: SchemaRemoteServer;
    close: () => void;
};

export const RemoteServerForm = ({ remoteServer, close }: RemoteServerFormProps) => {
    const [name, setName] = useState(remoteServer?.name ?? '');
    const [sshHostAlias, setSshHostAlias] = useState<string | undefined>(remoteServer?.ssh_host_alias);
    const [deviceType, setDeviceType] = useState<(typeof DEVICE_TYPES)[number] | undefined>(
        remoteServer && isSshDeviceType(remoteServer.device_type) ? remoteServer.device_type : undefined
    );
    const isEditing = remoteServer !== undefined;
    const { save, isPending, error } = useRemoteServerFormMutation(remoteServer);

    const { data: aliasOptions = [] } = $api.useQuery('get', '/api/remote-servers/aliases');
    const resolvedAlias = useMemo(
        () => aliasOptions.find((option) => option.alias === sshHostAlias),
        [aliasOptions, sshHostAlias]
    );

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (sshHostAlias === undefined || deviceType === undefined) {
            return;
        }

        save({ name: name.trim(), ssh_host_alias: sshHostAlias, device_type: deviceType }, { onSuccess: close });
    };

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
                        <TextField
                            // eslint-disable-next-line jsx-a11y/no-autofocus
                            autoFocus
                            isRequired
                            label='Name'
                            value={name}
                            onChange={setName}
                            width='100%'
                        />
                        <Picker
                            label='SSH host alias'
                            description='A Host stanza from your ~/.ssh/config.'
                            selectedKey={sshHostAlias ?? null}
                            onSelectionChange={(key) => setSshHostAlias(key ? String(key) : undefined)}
                            width='100%'
                            items={aliasOptions}
                            isRequired
                        >
                            {(option) => <Item key={option.alias}>{option.alias}</Item>}
                        </Picker>
                        {resolvedAlias !== undefined && (
                            <dl className={classes.resolvedHost}>
                                <dt>Resolves to</dt>
                                <dd>
                                    {[resolvedAlias.user, resolvedAlias.hostname].filter(Boolean).join('@') || '—'}
                                    {resolvedAlias.port ? `:${resolvedAlias.port}` : ''}
                                </dd>
                            </dl>
                        )}
                        <Picker
                            label='Device type'
                            description='Determines which trainer image is provisioned.'
                            selectedKey={deviceType ?? null}
                            onSelectionChange={(key) =>
                                setDeviceType(key ? (String(key) as (typeof DEVICE_TYPES)[number]) : undefined)
                            }
                            width='100%'
                            isRequired
                        >
                            {DEVICE_TYPES.map((type) => (
                                <Item key={type}>{type.toUpperCase()}</Item>
                            ))}
                        </Picker>
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
