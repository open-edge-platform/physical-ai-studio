import { FormEvent, useEffect, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Flex,
    Form,
    Grid,
    Heading,
    minmax,
    ProgressCircle,
    repeat,
    Text,
    TextField,
    View,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaRemoteTrainer } from '../../../api/openapi-spec';
import { IssueErrorAlert } from '../../../components/notification/issue-toasts';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';

import classes from './training-target-form.module.css';

const environmentDescription =
    'An existing environment in your Modal workspace. Environments isolate apps, storage Volumes, and Secrets ' +
    '(for example, development and production). Keep main unless you have created another environment in Modal. ' +
    'This is not a Python or robot environment.';

interface ModalProviderFormProps {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
}

export const ModalProviderForm = ({ remoteTrainer, close }: ModalProviderFormProps) => {
    const existing = remoteTrainer?.connection.connection_mode === 'modal' ? remoteTrainer.connection : undefined;
    const [name, setName] = useState(remoteTrainer?.name ?? '');
    const [appName, setAppName] = useState(existing?.app_name ?? 'physicalai-trainer');
    const [functionName, setFunctionName] = useState(existing?.function_name ?? 'train');
    const [volumeName, setVolumeName] = useState(existing?.volume_name ?? 'physicalai-training');
    const [environmentName, setEnvironmentName] = useState(existing?.environment_name ?? 'main');
    const [hfSecretName, setHfSecretName] = useState('');
    const [deployment, setDeployment] = useState<SchemaRemoteTrainer>();
    const update = $api.useMutation('patch', '/api/remote-trainers/{remote_trainer_id}', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });
    const deploy = $api.useMutation('post', '/api/remote-trainers/providers/modal/deploy', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });
    const health = $api.useQuery(
        'get',
        '/api/remote-trainers/{remote_trainer_id}/health',
        { params: { path: { remote_trainer_id: deployment?.id ?? '' } } },
        {
            enabled: deployment !== undefined,
            refetchInterval: (query) =>
                query.state.data?.status === 'healthy' || query.state.data?.status === 'degraded' ? false : 2000,
        }
    );
    useEffect(() => {
        if (health.data?.status === 'healthy') close();
    }, [health.data?.status, close]);
    const isPending = update.isPending || deploy.isPending;
    const error = update.error ?? deploy.error ?? health.error;
    const resourceNameIsValid = (value: string) => /^[\w-]{1,64}$/.test(value.trim());
    const isValid =
        name.trim().length > 0 &&
        name.trim().length <= 255 &&
        resourceNameIsValid(environmentName) &&
        (remoteTrainer
            ? [appName, functionName, volumeName].every(resourceNameIsValid)
            : !hfSecretName.trim() || resourceNameIsValid(hfSecretName));

    const submit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!isValid || isPending || deployment) return;
        try {
            if (!remoteTrainer) {
                const trainer = await deploy.mutateAsync({
                    body: {
                        name: name.trim(),
                        environment_name: environmentName.trim(),
                        hf_secret_name: hfSecretName.trim() || null,
                        confirm_deployment: true,
                    },
                });
                setDeployment(trainer);
                return;
            }
            const body = {
                name: name.trim(),
                connection: {
                    connection_mode: 'modal' as const,
                    app_name: appName.trim(),
                    function_name: functionName.trim(),
                    volume_name: volumeName.trim(),
                    environment_name: environmentName.trim(),
                },
            };
            await update.mutateAsync({ params: { path: { remote_trainer_id: remoteTrainer.id } }, body });
            close();
        } catch {
            return;
        }
    };

    return (
        <Form onSubmit={submit} validationBehavior='native'>
            <Dialog width='size-6000' maxHeight='85vh'>
                <Heading>{remoteTrainer ? 'Edit Modal training target' : 'Connect to Modal'}</Heading>
                <Divider />
                <Content>
                    <Flex direction='column' gap='size-250'>
                        {deployment ? (
                            <>
                                {health.data?.status === 'degraded' ? (
                                    <InlineAlert variant='error'>
                                        {health.data.issues?.[0]?.detail ?? health.data.reason_code}
                                    </InlineAlert>
                                ) : (
                                    <Flex gap='size-150' alignItems='center'>
                                        <ProgressCircle aria-label='Deploying Modal trainer' isIndeterminate size='S' />
                                        <div role='status' aria-live='polite'>
                                            {health.data?.reason_code ?? 'Preparing your Modal trainer…'}
                                        </div>
                                    </Flex>
                                )}
                                <Text>
                                    Your target has been saved. You can close this dialog and follow its status in the
                                    training targets list. Closing does not stop deployment or delete cloud resources.
                                </Text>
                            </>
                        ) : (
                            <>
                                <Text>
                                    {remoteTrainer
                                        ? 'Update your Modal connection details. This does not redeploy the trainer.'
                                        : 'Studio deploys a new trainer and connects it automatically. ' +
                                          'Select the GPU when starting a training job.'}
                                </Text>
                                <TextField
                                    label='Name'
                                    placeholder='e.g. Modal'
                                    value={name}
                                    onChange={setName}
                                    isRequired
                                    isDisabled={isPending}
                                    maxLength={255}
                                    width='100%'
                                />
                                {!remoteTrainer ? (
                                    <>
                                        <Disclosure isQuiet>
                                            <DisclosureTitle UNSAFE_className={classes.disclosureTitle}>
                                                Advanced settings
                                            </DisclosureTitle>
                                            <DisclosurePanel UNSAFE_style={{ paddingInline: 0 }}>
                                                <Flex direction='column' gap='size-150'>
                                                    <TextField
                                                        label='Environment'
                                                        description={environmentDescription}
                                                        value={environmentName}
                                                        onChange={setEnvironmentName}
                                                        isRequired
                                                        isDisabled={isPending}
                                                        width='100%'
                                                    />
                                                    <TextField
                                                        label='Hugging Face secret (optional)'
                                                        value={hfSecretName}
                                                        onChange={setHfSecretName}
                                                        isDisabled={isPending}
                                                        description={
                                                            <>
                                                                Existing Modal secret with{' '}
                                                                <code className={classes.inlineCode}>HF_TOKEN</code>,
                                                                used when no token is configured in Studio Settings.
                                                            </>
                                                        }
                                                        width='100%'
                                                    />
                                                </Flex>
                                            </DisclosurePanel>
                                        </Disclosure>
                                    </>
                                ) : (
                                    <View backgroundColor='gray-75' padding='size-200' borderRadius='regular'>
                                        <Flex direction='column' gap='size-150'>
                                            <Heading level={4} margin={0}>
                                                Modal resources
                                            </Heading>
                                            <Grid
                                                columns={repeat('auto-fit', minmax('size-2000', '1fr'))}
                                                gap='size-150'
                                            >
                                                <TextField
                                                    label='App name'
                                                    value={appName}
                                                    onChange={setAppName}
                                                    isRequired
                                                    isDisabled={isPending}
                                                    width='100%'
                                                />
                                                <TextField
                                                    label='Environment'
                                                    description={environmentDescription}
                                                    value={environmentName}
                                                    onChange={setEnvironmentName}
                                                    isRequired
                                                    isDisabled={isPending}
                                                    width='100%'
                                                />
                                                <TextField
                                                    label='Function name'
                                                    value={functionName}
                                                    onChange={setFunctionName}
                                                    isRequired
                                                    isDisabled={isPending}
                                                    width='100%'
                                                />
                                                <TextField
                                                    label='Volume name'
                                                    value={volumeName}
                                                    onChange={setVolumeName}
                                                    isRequired
                                                    isDisabled={isPending}
                                                    width='100%'
                                                />
                                            </Grid>
                                        </Flex>
                                    </View>
                                )}
                                <Disclosure isQuiet>
                                    <DisclosureTitle UNSAFE_className={classes.disclosureTitle}>
                                        Deployment &amp; credentials
                                    </DisclosureTitle>
                                    <DisclosurePanel UNSAFE_style={{ paddingInline: 0 }}>
                                        <Flex direction='column' gap='size-150'>
                                            <Text>
                                                Sign in with <code className={classes.inlineCode}>modal setup</code> on
                                                the Studio backend, or set{' '}
                                                <code className={classes.inlineCode}>MODAL_TOKEN_ID</code> and{' '}
                                                <code className={classes.inlineCode}>MODAL_TOKEN_SECRET</code> in its
                                                environment. Credentials are managed on the backend, not in this form.
                                            </Text>
                                        </Flex>
                                    </DisclosurePanel>
                                </Disclosure>
                                {!remoteTrainer && (
                                    <Text>
                                        Deploy and connect creates a new Modal app and storage Volume. Cloud charges may
                                        apply.
                                    </Text>
                                )}
                            </>
                        )}
                        {error && <IssueErrorAlert error={error} />}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        {deployment ? 'Close' : 'Cancel'}
                    </Button>
                    {!deployment && (
                        <Button variant='accent' type='submit' isDisabled={!isValid || isPending} isPending={isPending}>
                            {remoteTrainer ? 'Save changes' : 'Deploy and connect'}
                        </Button>
                    )}
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
