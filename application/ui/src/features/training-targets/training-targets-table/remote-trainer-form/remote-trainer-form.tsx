import { FormEvent, useState } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Form,
    Heading,
    Item,
    Link,
    NumberField,
    Picker,
    TabList,
    Tabs,
    Text,
    TextArea,
    TextField,
} from '@geti-ui/ui';
import { Add, ExternalLinkIcon } from '@geti-ui/ui/icons';

import { getApiErrorMessage, getSshHostKeyFingerprint } from '../../../../api/errors';
import { SchemaAwsBatchConnection, SchemaRemoteTrainer } from '../../../../api/openapi-spec';
import { ReactComponent as AwsIcon } from '../../../../assets/icons/aws-icon.svg';
import { AddSshHostDialog } from '../add-ssh-host-dialog';
import { SshHostKeyConfirmation } from '../ssh-host-key-confirmation-dialog';
import { parseAwsBatchConfiguration } from './aws-batch-configuration';
import { INSECURE_TRAINER_URL_WARNING, isInsecureTrainerUrl } from './insecure-trainer-url';
import { InfoHelp } from './ssh-tunnel-section';
import { RemoteTrainerFormValues, useRemoteTrainerFormMutation } from './use-remote-trainer-form-mutation';
import { useSshHostAliases } from './use-ssh-host-aliases';

import classes from './remote-trainer-form.module.css';

const SSH_HOST_SETUP_URL =
    'https://github.com/open-edge-platform/physical-ai-studio/blob/main/' +
    'application/docs/07-remote-training.md#prepare-an-ssh-host';
const AWS_TEMPLATE_BASE = 'https://physical-ai-studio.s3.eu-west-1.amazonaws.com/aws-cf-templates/';
const awsStackUrl = (template: string, stackName: string) =>
    'https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1' +
    `#/stacks/create/review?templateURL=${encodeURIComponent(AWS_TEMPLATE_BASE + template)}` +
    `&stackName=${stackName}`;
const AWS_EC2_STACK_URL = awsStackUrl('remote-trainer.yaml', 'physical-ai-studio-remote-trainer');
const AWS_BATCH_STACK_URL = awsStackUrl('aws-batch-trainer.yaml', 'physical-ai-studio-batch-trainer');

type RemoteTrainerFormProps = {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
    requestHostKeyConfirmation: (confirmation: SshHostKeyConfirmation) => void;
    sshAvailable?: boolean;
};

type ConnectionMode = SchemaRemoteTrainer['connection_mode'];
type SshHostSource = 'manual' | 'pick';

export const RemoteTrainerForm = ({
    remoteTrainer,
    close,
    requestHostKeyConfirmation,
    sshAvailable = true,
}: RemoteTrainerFormProps) => {
    const existing = remoteTrainer?.connection;
    const existingSsh = existing?.connection_mode === 'ssh' ? existing : undefined;
    const existingDirect = existing?.connection_mode === 'direct' ? existing : undefined;
    const existingAws = existing?.connection_mode === 'aws_batch' ? existing : undefined;

    const [name, setName] = useState(remoteTrainer?.name ?? '');
    const [url, setUrl] = useState(existingDirect?.url ?? '');
    const [connectionMode, setConnectionMode] = useState<ConnectionMode>(
        existing?.connection_mode ?? (sshAvailable ? 'ssh' : 'direct')
    );
    const [sshHostSource, setSshHostSource] = useState<SshHostSource>(existingSsh?.ssh_connection ? 'manual' : 'pick');
    const [sshHostAlias, setSshHostAlias] = useState(existingSsh?.ssh_host_alias ?? '');
    const [sshHostname, setSshHostname] = useState(existingSsh?.ssh_connection?.hostname ?? '');
    const [sshPort, setSshPort] = useState<number | undefined>(existingSsh?.ssh_connection?.port ?? 22);
    const [sshUser, setSshUser] = useState(
        remoteTrainer === undefined ? 'ec2-user' : (existingSsh?.ssh_connection?.user ?? '')
    );
    const [sshIdentityFile, setSshIdentityFile] = useState(existingSsh?.ssh_connection?.identity_file ?? '');
    const [sshRemotePort, setSshRemotePort] = useState<number | undefined>(existingSsh?.ssh_remote_port ?? 8001);
    const [sshLocalPort, setSshLocalPort] = useState<number | undefined>(existingSsh?.ssh_local_port ?? 8001);
    const [awsConfigText, setAwsConfigText] = useState(
        existingAws ? JSON.stringify(stripMode(existingAws), null, 2) : ''
    );
    const [installPrerequisites, setInstallPrerequisites] = useState(false);
    const isEditing = remoteTrainer !== undefined;
    const { aliases } = useSshHostAliases(sshAvailable);
    const { save, reset, isPending, error } = useRemoteTrainerFormMutation(remoteTrainer);

    const isSsh = connectionMode === 'ssh';
    const isAws = connectionMode === 'aws_batch';
    const isManual = isSsh && sshHostSource === 'manual';
    const awsConfig = isAws ? parseAwsBatchConfiguration(awsConfigText) : undefined;

    const connection: RemoteTrainerFormValues['connection'] | undefined = isSsh
        ? {
              connection_mode: 'ssh',
              ssh_host_alias: isManual ? null : sshHostAlias.trim(),
              ssh_connection: isManual
                  ? {
                        hostname: sshHostname.trim(),
                        port: sshPort ?? 22,
                        user: sshUser.trim() || null,
                        identity_file: sshIdentityFile.trim() || null,
                    }
                  : null,
              ssh_remote_port: sshRemotePort ?? 8001,
              ssh_local_port: sshLocalPort ?? 8001,
          }
        : isAws
          ? awsConfig?.value
          : { connection_mode: 'direct', url };
    const values: RemoteTrainerFormValues | undefined =
        connection === undefined ? undefined : { name: name.trim(), connection };

    const sshHostIdentity = isManual ? `${sshHostname.trim()}:${sshPort ?? 22}` : sshHostAlias.trim();
    const errorMessage =
        error && getSshHostKeyFingerprint(error) === undefined
            ? (getApiErrorMessage(error) ?? 'The remote trainer could not be saved. Try again.')
            : undefined;

    const requestConfirmation = (fingerprint: string) => {
        requestHostKeyConfirmation({
            fingerprint,
            host: sshHostIdentity,
            onConfirm: () => submit(fingerprint),
            onCancel: reset,
        });
    };

    const submit = (acceptedHostKeyFingerprint?: string) => {
        if (values === undefined) return;
        save(values, {
            onSuccess: close,
            installPrerequisites: !isEditing && isSsh && installPrerequisites,
            acceptedHostKeyFingerprint,
            onHostKeyConfirmationRequired: requestConfirmation,
        });
    };

    const hasValidSshHost = isManual ? sshHostname.trim() !== '' : sshHostAlias.trim() !== '';
    const canSubmit =
        name.trim() !== '' &&
        (isSsh
            ? sshAvailable && hasValidSshHost && Boolean(sshRemotePort) && Boolean(sshLocalPort)
            : isAws
              ? awsConfig?.value !== undefined
              : url.trim() !== '');

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (canSubmit) {
            submit();
        }
    };

    const isInsecureUrl = isInsecureTrainerUrl(url);

    return (
        <Form onSubmit={handleSubmit} validationBehavior='native'>
            <Dialog width='size-6000'>
                <Heading>{isEditing ? 'Edit remote trainer' : 'Add remote trainer'}</Heading>
                <Divider />
                <Content>
                    <Flex direction='column' gap='size-150'>
                        <TextField
                            // eslint-disable-next-line jsx-a11y/no-autofocus
                            autoFocus
                            isRequired
                            label='Name'
                            value={name}
                            onChange={setName}
                            width='100%'
                        />
                        <Tabs
                            selectedKey={connectionMode}
                            disabledKeys={!sshAvailable ? ['ssh'] : []}
                            onSelectionChange={(key) => setConnectionMode(key as ConnectionMode)}
                        >
                            <TabList aria-label='Connection method'>
                                <Item key='ssh'>SSH tunnel</Item>
                                <Item key='direct'>Trainer URL</Item>
                                <Item key='aws_batch'>AWS Batch</Item>
                            </TabList>
                        </Tabs>
                        <div className={classes.modeFields}>
                            <Text>
                                {isSsh
                                    ? 'Connect through SSH when the trainer is not directly reachable.'
                                    : isAws
                                      ? 'Run each training as an AWS Batch job on compute provisioned on demand.'
                                      : 'Enter the URL of a trainer that Studio can reach directly.'}
                            </Text>
                            {isAws && (
                                <>
                                    <TextArea
                                        isRequired
                                        label='Studio configuration'
                                        value={awsConfigText}
                                        onChange={setAwsConfigText}
                                        description='Paste the StudioConfiguration output of the AWS Batch stack.'
                                        validationState={
                                            awsConfigText.trim() !== '' && awsConfig?.error ? 'invalid' : undefined
                                        }
                                        errorMessage={awsConfig?.error}
                                        contextualHelp={
                                            <InfoHelp title='Studio configuration'>
                                                After the CloudFormation stack finishes, open its Outputs tab and copy
                                                the StudioConfiguration value here.
                                            </InfoHelp>
                                        }
                                        width='100%'
                                    />
                                    {!isEditing && (
                                        <Link href={AWS_BATCH_STACK_URL} target='_blank' rel='noopener noreferrer'>
                                            <span className={classes.awsStackLinkContent}>
                                                Deploy AWS Batch stack
                                                <ExternalLinkIcon aria-hidden='true' />
                                            </span>
                                        </Link>
                                    )}
                                </>
                            )}
                            {connectionMode === 'direct' ? (
                                <TextField
                                    isRequired
                                    label='Trainer URL'
                                    type='url'
                                    value={url}
                                    onChange={setUrl}
                                    description='Address exposed by the remote trainer.'
                                    contextualHelp={
                                        <InfoHelp title='Trainer URL'>
                                            Use the complete endpoint URL, including its scheme and port.
                                        </InfoHelp>
                                    }
                                    width='100%'
                                />
                            ) : null}
                            {connectionMode === 'direct' && isInsecureUrl && (
                                <Text UNSAFE_className={classes.errorMessage}>{INSECURE_TRAINER_URL_WARNING}</Text>
                            )}
                            {isSsh && !sshAvailable && <Text>SSH is unavailable in this environment.</Text>}
                            {isSsh && sshAvailable && (
                                <Flex direction='column' gap='size-100'>
                                    <div className={classes.fieldRow}>
                                        <NumberField
                                            isRequired
                                            label='Remote port'
                                            value={sshRemotePort}
                                            onChange={setSshRemotePort}
                                            minValue={1}
                                            maxValue={65535}
                                            formatOptions={{ useGrouping: false }}
                                            contextualHelp={
                                                <InfoHelp title='Remote port'>
                                                    Port where the trainer listens remotely.
                                                </InfoHelp>
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
                                            formatOptions={{ useGrouping: false }}
                                            contextualHelp={
                                                <InfoHelp title='Local port'>
                                                    Loopback port the tunnel binds to on this Studio host.
                                                </InfoHelp>
                                            }
                                            width='100%'
                                        />
                                    </div>
                                    <Tabs
                                        selectedKey={sshHostSource}
                                        onSelectionChange={(key) => setSshHostSource(key as SshHostSource)}
                                    >
                                        <TabList aria-label='SSH connection'>
                                            <Item key='pick'>Config alias</Item>
                                            <Item key='manual'>Connection details</Item>
                                        </TabList>
                                    </Tabs>
                                    {sshHostSource === 'manual' ? (
                                        <>
                                            <div className={classes.fieldRow}>
                                                <TextField
                                                    isRequired
                                                    label='Host'
                                                    value={sshHostname}
                                                    onChange={setSshHostname}
                                                    contextualHelp={
                                                        <InfoHelp title='Host'>
                                                            Hostname or IP address to connect to.
                                                        </InfoHelp>
                                                    }
                                                    width='100%'
                                                />
                                                <NumberField
                                                    label='Port'
                                                    value={sshPort}
                                                    onChange={setSshPort}
                                                    minValue={1}
                                                    maxValue={65535}
                                                    width='100%'
                                                />
                                            </div>
                                            <div className={classes.fieldRow}>
                                                <TextField
                                                    label='User'
                                                    value={sshUser}
                                                    onChange={setSshUser}
                                                    width='100%'
                                                />
                                                <TextField
                                                    label='Key path'
                                                    value={sshIdentityFile}
                                                    onChange={setSshIdentityFile}
                                                    contextualHelp={
                                                        <InfoHelp title='Key path'>
                                                            Path to a private key file on this Studio host.
                                                        </InfoHelp>
                                                    }
                                                    width='100%'
                                                />
                                            </div>
                                        </>
                                    ) : (
                                        <Flex alignItems='end' gap='size-100'>
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
                                            <DialogTrigger>
                                                <ActionButton aria-label='Add SSH connection'>
                                                    <Add />
                                                </ActionButton>
                                                {(closeAddHostDialog) => (
                                                    <AddSshHostDialog
                                                        close={closeAddHostDialog}
                                                        onCreated={(option) => setSshHostAlias(option.alias)}
                                                    />
                                                )}
                                            </DialogTrigger>
                                        </Flex>
                                    )}
                                    {!isEditing && (
                                        <Flex alignItems='center' gap='size-50'>
                                            <Checkbox
                                                isSelected={installPrerequisites}
                                                onChange={setInstallPrerequisites}
                                            >
                                                Set up Docker and GPU support
                                            </Checkbox>
                                            <InfoHelp title='SSH host setup'>
                                                On Ubuntu 24.04 or 26.04, Studio checks the selected SSH host and
                                                installs missing Docker and NVIDIA or Intel GPU packages before pulling
                                                the trainer image. Missing prerequisites require passwordless sudo. A
                                                reboot needs separate confirmation; SSH re-login may also be needed.
                                                Docker access grants root-equivalent privileges.{' '}
                                                <Link
                                                    href={SSH_HOST_SETUP_URL}
                                                    target='_blank'
                                                    rel='noopener noreferrer'
                                                >
                                                    How to prepare an SSH host
                                                </Link>
                                            </InfoHelp>
                                        </Flex>
                                    )}
                                </Flex>
                            )}
                        </div>
                        {errorMessage !== undefined && (
                            <Text UNSAFE_className={classes.errorMessage}>{errorMessage}</Text>
                        )}
                        {!isEditing && !isAws && (
                            <>
                                <Divider size='S' />
                                <Flex direction='column' alignItems='start' gap='size-50'>
                                    <AwsIcon aria-hidden='true' className={classes.awsIcon} />
                                    <Flex alignItems='center' gap='size-75' UNSAFE_className={classes.awsPrompt}>
                                        <Text>Need a new remote trainer?</Text>
                                        <Link href={AWS_EC2_STACK_URL} target='_blank' rel='noopener noreferrer'>
                                            <span className={classes.awsStackLinkContent}>
                                                Deploy AWS stack
                                                <ExternalLinkIcon aria-hidden='true' />
                                            </span>
                                        </Link>
                                    </Flex>
                                </Flex>
                            </>
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

const stripMode = ({ connection_mode: _mode, ...rest }: SchemaAwsBatchConnection) => rest;
