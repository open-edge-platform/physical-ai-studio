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
    ToggleButtons,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaRemoteServer } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { useRemoteServerFormMutation } from '../training-targets-table/remote-server-form/use-remote-server-form-mutation';
import { VerifyAfterSaveDialog } from '../training-targets-table/remote-server-form/verify-after-save-dialog';
import { useRemoteTrainerFormMutation } from '../training-targets-table/remote-trainer-form/use-remote-trainer-form-mutation';

import classes from './training-target-form.module.css';

type TargetType = 'ssh' | 'direct-url';
const TARGET_TYPES: TargetType[] = ['ssh', 'direct-url'];

const TARGET_TYPE_LABELS: Record<TargetType, string> = {
    ssh: 'SSH provisioned',
    'direct-url': 'Direct trainer URL',
};

const DEVICE_TYPES = ['cuda', 'xpu'] as const;
type DeviceType = (typeof DEVICE_TYPES)[number];

type TrainingTargetFormProps = {
    close: () => void;
};

/**
 * Unified "New training target" entry point. A single dialog with a segmented
 * type switch at the top, rather than a menu that opens one of two separate
 * dialogs — the two target kinds share the Name field and this shell, and
 * only the fields below the switch change.
 */
export const TrainingTargetForm = ({ close }: TrainingTargetFormProps) => {
    const [targetType, setTargetType] = useState<TargetType>('ssh');
    const [name, setName] = useState('');
    const [url, setUrl] = useState('');
    const [sshHostAlias, setSshHostAlias] = useState<string | undefined>(undefined);
    const [deviceType, setDeviceType] = useState<DeviceType | undefined>(undefined);
    // Set once an SSH server is saved, to swap this dialog for the "pull &
    // verify now?" prompt (see `VerifyAfterSaveDialog`). A direct-URL trainer
    // has no such preflight, so it always just closes on save.
    const [savedServer, setSavedServer] = useState<SchemaRemoteServer | undefined>(undefined);

    const {
        save: saveTrainer,
        isPending: isSavingTrainer,
        error: trainerError,
    } = useRemoteTrainerFormMutation(undefined);
    const { save: saveServer, isPending: isSavingServer, error: serverError } = useRemoteServerFormMutation(undefined);

    const { data: aliasOptions = [] } = $api.useQuery('get', '/api/remote-servers/aliases', undefined, {
        enabled: targetType === 'ssh',
    });
    const resolvedAlias = useMemo(
        () => aliasOptions.find((option) => option.alias === sshHostAlias),
        [aliasOptions, sshHostAlias]
    );

    const isPending = targetType === 'ssh' ? isSavingServer : isSavingTrainer;
    const error = targetType === 'ssh' ? serverError : trainerError;
    const errorMessage = error
        ? (getApiErrorMessage(error) ?? 'The training target could not be saved. Try again.')
        : undefined;

    const canSubmit =
        name.trim() !== '' &&
        (targetType === 'ssh' ? sshHostAlias !== undefined && deviceType !== undefined : url !== '');

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (targetType === 'ssh') {
            if (sshHostAlias === undefined || deviceType === undefined) {
                return;
            }
            saveServer(
                { name: name.trim(), ssh_host_alias: sshHostAlias, device_type: deviceType },
                { onSuccess: setSavedServer }
            );
            return;
        }

        saveTrainer({ name: name.trim(), url }, { onSuccess: close });
    };

    if (savedServer !== undefined) {
        return <VerifyAfterSaveDialog savedServer={savedServer} close={close} />;
    }

    return (
        <Form onSubmit={handleSubmit} validationBehavior='native' width='size-6000'>
            <Dialog>
                <Heading>Add training target</Heading>
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

                        <Flex direction='column' gap='size-75'>
                            <Text UNSAFE_className={classes.toggleLabel}>Target type</Text>
                            <ToggleButtons
                                options={TARGET_TYPES}
                                selectedOption={targetType}
                                onOptionChange={setTargetType}
                                getLabel={(option) => TARGET_TYPE_LABELS[option]}
                            />
                            <Text UNSAFE_className={classes.hint}>
                                SSH targets launch a trainer container per job. Direct endpoints run an already-managed
                                trainer.
                            </Text>
                        </Flex>

                        {targetType === 'ssh' ? (
                            <>
                                <Picker
                                    label='SSH host'
                                    description='Pick a Host entry from your ~/.ssh/config.'
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
                                            {[resolvedAlias.user, resolvedAlias.hostname].filter(Boolean).join('@') ||
                                                '—'}
                                            {resolvedAlias.port ? `:${resolvedAlias.port}` : ''}
                                        </dd>
                                    </dl>
                                )}
                                <Picker
                                    label='Device type'
                                    description='Determines which trainer image is provisioned.'
                                    selectedKey={deviceType ?? null}
                                    onSelectionChange={(key) =>
                                        setDeviceType(key ? (String(key) as DeviceType) : undefined)
                                    }
                                    width='100%'
                                    isRequired
                                >
                                    {DEVICE_TYPES.map((type) => (
                                        <Item key={type}>{type.toUpperCase()}</Item>
                                    ))}
                                </Picker>
                            </>
                        ) : (
                            <TextField
                                isRequired
                                label='Trainer URL'
                                type='url'
                                value={url}
                                onChange={setUrl}
                                description='Use the endpoint URL that accepts Physical AI Studio training jobs.'
                                width='100%'
                            />
                        )}

                        {errorMessage !== undefined && <InlineAlert variant='error'>{errorMessage}</InlineAlert>}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        Cancel
                    </Button>
                    <Button variant='accent' type='submit' isDisabled={!canSubmit} isPending={isPending}>
                        {targetType === 'ssh' ? 'Verify & save' : 'Add trainer'}
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
