import { FormEvent, useEffect, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Form,
    Heading,
    Text,
    TextField,
    ToggleButtons,
} from '@geti-ui/ui';

import { getApiErrorMessage } from '../../../api/errors';
import { SchemaRemoteServer } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { SshDeviceType, SshTargetFields } from '../training-targets-table/remote-server-form/ssh-target-fields';
import { useRemoteServerFormMutation } from '../training-targets-table/remote-server-form/use-remote-server-form-mutation';
import { VerifyAfterSaveDialog } from '../training-targets-table/remote-server-form/verify-after-save-dialog';
import { useRemoteTrainerFormMutation } from '../training-targets-table/remote-trainer-form/use-remote-trainer-form-mutation';
import { useDeviceTypeDetection } from './use-device-type-detection';

import classes from './training-target-form.module.css';

type TargetType = 'ssh' | 'direct-url';
const TARGET_TYPES: TargetType[] = ['ssh', 'direct-url'];

const TARGET_TYPE_LABELS: Record<TargetType, string> = {
    ssh: 'SSH provisioned',
    'direct-url': 'Direct trainer URL',
};

const DEVICE_TYPE_DESCRIPTION = 'Determines which trainer image is provisioned.';

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
    const [deviceType, setDeviceType] = useState<SshDeviceType | undefined>(undefined);
    // True once the user picks a device type themselves, so a later
    // autodetection response (e.g. from switching SSH hosts back and forth)
    // never clobbers a deliberate choice. Reset whenever the host alias
    // changes, so picking a new host re-enables autodetection for it.
    const [deviceTypeTouched, setDeviceTypeTouched] = useState(false);
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

    const { detectedDeviceType, isDetecting } = useDeviceTypeDetection(targetType === 'ssh' ? sshHostAlias : undefined);

    useEffect(() => {
        // A new host alias means a new device to detect, and the previous
        // detection (or manual pick) no longer applies to it.
        setDeviceTypeTouched(false);
        setDeviceType(undefined);
    }, [sshHostAlias]);

    useEffect(() => {
        if (!deviceTypeTouched && detectedDeviceType !== undefined) {
            setDeviceType(detectedDeviceType);
        }
    }, [detectedDeviceType, deviceTypeTouched]);

    const deviceTypeDescription = isDetecting
        ? 'Detecting the accelerator on this host…'
        : !deviceTypeTouched && detectedDeviceType !== undefined
          ? 'Auto-detected from the SSH host. Change it if this is wrong.'
          : DEVICE_TYPE_DESCRIPTION;

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
                            <SshTargetFields
                                sshHostAlias={sshHostAlias}
                                onSshHostAliasChange={setSshHostAlias}
                                deviceType={deviceType}
                                onDeviceTypeChange={(value) => {
                                    setDeviceTypeTouched(true);
                                    setDeviceType(value);
                                }}
                                deviceTypeDescription={deviceTypeDescription}
                            />
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
