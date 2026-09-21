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
import { RemoteTrainerForm } from '../training-targets-table/remote-trainer-form/remote-trainer-form';
import { SshHostKeyConfirmation } from '../training-targets-table/ssh-host-key-confirmation-dialog';
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
    requestHostKeyConfirmation: (confirmation: SshHostKeyConfirmation) => void;
    // Whether the SSH-provisioned target type is selectable at all. The
    // backend fails closed on every SSH-provisioned route whenever this
    // Studio instance is not eligible to run the feature (e.g. it is bound to
    // more than loopback, see `training-targets-page.tsx`), so the type
    // switch below only makes sense to show when there is an actual choice
    // to make. When SSH is unavailable, this form always behaves as the
    // direct-URL trainer form.
    sshAvailable: boolean;
};

/**
 * Unified "New training target" entry point. A single dialog with a segmented
 * type switch at the top, rather than a menu that opens one of two separate
 * dialogs. The "SSH provisioned" kind is handled entirely here; the "Direct
 * trainer URL" kind delegates to `RemoteTrainerForm` (which already covers a
 * direct URL, an SSH-tunneled trainer, and the "Deploy AWS stack" prompt) so
 * that richer form isn't duplicated for the create flow.
 */
export const TrainingTargetForm = ({ close, requestHostKeyConfirmation, sshAvailable }: TrainingTargetFormProps) => {
    const [targetType, setTargetType] = useState<TargetType>(sshAvailable ? 'ssh' : 'direct-url');
    const [name, setName] = useState('');
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

    const { save: saveServer, isPending: isSavingServer, error: serverError } = useRemoteServerFormMutation(undefined);

    const { detectedDeviceType, isDetecting } = useDeviceTypeDetection(targetType === 'ssh' ? sshHostAlias : undefined);

    useEffect(() => {
        // A new host alias means a new device to detect, and the previous
        // detection (or manual pick) no longer applies to it.
        setDeviceTypeTouched(false);
        setDeviceType(undefined);
    }, [sshHostAlias]);

    useEffect(() => {
        // SSH availability can flip after the dialog is already open (the
        // `/api/remote-servers` query resolving to `ssh_feature_unavailable`
        // races the dialog opening on first mount) - fall back to the
        // direct-URL fields rather than leaving `targetType` stuck on 'ssh'
        // with no way to submit.
        if (!sshAvailable && targetType === 'ssh') {
            setTargetType('direct-url');
        }
    }, [sshAvailable, targetType]);

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

    const errorMessage = serverError
        ? (getApiErrorMessage(serverError) ?? 'The training target could not be saved. Try again.')
        : undefined;

    const canSubmit = name.trim() !== '' && sshHostAlias !== undefined && deviceType !== undefined;

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (sshHostAlias === undefined || deviceType === undefined) {
            return;
        }
        saveServer(
            { name: name.trim(), ssh_host_alias: sshHostAlias, device_type: deviceType },
            { onSuccess: setSavedServer }
        );
    };

    if (savedServer !== undefined) {
        return <VerifyAfterSaveDialog savedServer={savedServer} close={close} />;
    }

    const typeSwitch = sshAvailable && (
        <Flex direction='column' gap='size-75'>
            <Text UNSAFE_className={classes.toggleLabel}>Target type</Text>
            <ToggleButtons
                options={TARGET_TYPES}
                selectedOption={targetType}
                onOptionChange={setTargetType}
                getLabel={(option) => TARGET_TYPE_LABELS[option]}
            />
            <Text UNSAFE_className={classes.hint}>
                SSH targets launch a trainer container per job. Direct endpoints run an already-managed trainer.
            </Text>
        </Flex>
    );

    if (targetType === 'direct-url') {
        return (
            <RemoteTrainerForm
                close={close}
                requestHostKeyConfirmation={requestHostKeyConfirmation}
                typeSwitch={typeSwitch}
                initialName={name}
            />
        );
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
                        {typeSwitch}
                        <InlineAlert variant='warning'>
                            <strong>Security risk:</strong> SSH servers have no built-in authentication. Anyone who can
                            reach this Studio backend can run arbitrary code as root on the server. Only connect to
                            servers you trust, and only run Studio on a single-user, localhost-only workstation.
                        </InlineAlert>
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
                        {errorMessage !== undefined && <InlineAlert variant='error'>{errorMessage}</InlineAlert>}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isSavingServer}>
                        Cancel
                    </Button>
                    <Button variant='accent' type='submit' isDisabled={!canSubmit} isPending={isSavingServer}>
                        Verify & save
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
