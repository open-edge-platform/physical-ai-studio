import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Form, Heading, Link, Text, TextField } from '@geti-ui/ui';
import { ExternalLinkIcon } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { SchemaRemoteTrainer } from '../../../api/openapi-spec';
import { IssueErrorAlert } from '../../../components/notification/issue-toasts';
import { AwsCredentials } from './aws-credentials';

import classes from '../training-targets-table/remote-trainer-form/remote-trainer-form.module.css';

const AWS_STACK_URL =
    'https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1' +
    '#/stacks/create/review?templateURL=' +
    encodeURIComponent(
        'https://physical-ai-studio.s3.eu-west-1.amazonaws.com/aws-cf-templates/aws-batch-trainer/v1/aws-batch-trainer.yaml'
    ) +
    '&stackName=physical-ai-studio-batch-trainer';

interface CloudProviderFormProps {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
}

export const CloudProviderForm = ({ remoteTrainer, close }: CloudProviderFormProps) => {
    const existing = remoteTrainer?.connection.connection_mode === 'aws_batch' ? remoteTrainer.connection : undefined;
    const [step, setStep] = useState<'configure' | 'connect'>('configure');
    const [profileName, setProfileName] = useState(existing?.profile_name ?? '');
    const [values, setValues] = useState<Record<string, string>>({
        configuration_uri: existing?.configuration_uri ?? '',
    });
    const {
        data: providers,
        isPending: isLoadingSchema,
        isError: schemaError,
    } = $api.useQuery('get', '/api/remote-trainers/providers');
    const provider = providers?.find(({ id }) => id === 'aws');
    const resolve = $api.useMutation('post', '/api/remote-trainers/providers/aws/configuration');
    const create = $api.useMutation('post', '/api/remote-trainers', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });
    const update = $api.useMutation('patch', '/api/remote-trainers/{remote_trainer_id}', {
        meta: { invalidates: [['get', '/api/remote-trainers']] },
    });
    const isPending = resolve.isPending || create.isPending || update.isPending;
    const error = resolve.error ?? create.error ?? update.error;
    const isConfigurationValid =
        provider !== undefined &&
        provider.fields.every((field) => {
            const value = (values[field.name] ?? '').trim();
            return (
                value.length >= (field.minLength ?? 1) &&
                (field.maxLength == null || value.length <= field.maxLength) &&
                (field.pattern == null || new RegExp(field.pattern).test(value))
            );
        });

    const submit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!isConfigurationValid || isPending) return;
        if (step === 'configure') {
            setStep('connect');
            return;
        }
        if (!profileName.trim()) return;
        create.reset();
        update.reset();
        try {
            const connection = await resolve.mutateAsync({
                body: {
                    configuration_uri: values.configuration_uri.trim(),
                    profile_name: profileName.trim(),
                },
            });
            const body = { name: 'AWS Provider', connection };
            if (remoteTrainer) {
                await update.mutateAsync({ params: { path: { remote_trainer_id: remoteTrainer.id } }, body });
            } else {
                await create.mutateAsync({ body });
            }
            close();
        } catch {
            return;
        }
    };

    return (
        <Form onSubmit={submit} validationBehavior='native'>
            <Dialog width='size-6000' maxHeight='85vh'>
                <Heading>
                    {step === 'configure'
                        ? remoteTrainer
                            ? 'Configure AWS stack'
                            : 'Deploy and configure AWS stack'
                        : 'Connect to AWS remote training stack'}
                </Heading>
                <Divider />
                <Content>
                    <Flex direction='column' gap='size-150'>
                        {step === 'configure' && (
                            <>
                                {!remoteTrainer && (
                                    <>
                                        <Link href={AWS_STACK_URL} target='_blank' rel='noopener noreferrer'>
                                            <span className={classes.awsStackLinkContent}>
                                                Deploy AWS Batch stack
                                                <ExternalLinkIcon aria-hidden='true' />
                                            </span>
                                        </Link>
                                        <Text UNSAFE_className={classes.helperText}>
                                            Follow the link to provision the AWS remote training stack, or use an
                                            existing stack.
                                        </Text>
                                        <Divider size='S' />
                                    </>
                                )}
                                {isLoadingSchema && <Text>Loading configuration fields...</Text>}
                                {(schemaError || (!isLoadingSchema && !provider)) && (
                                    <div role='alert'>AWS configuration fields could not be loaded.</div>
                                )}
                                {provider?.fields.map((field) => (
                                    <TextField
                                        key={field.name}
                                        label={field.title}
                                        placeholder='s3://your-bucket/studio-config.json'
                                        description='Paste ConfigurationUri from the stack’s Outputs tab.'
                                        isRequired
                                        value={values[field.name] ?? ''}
                                        onChange={(value) =>
                                            setValues((current) => ({ ...current, [field.name]: value }))
                                        }
                                        minLength={field.minLength ?? undefined}
                                        maxLength={field.maxLength ?? undefined}
                                        pattern={field.pattern ?? undefined}
                                        isDisabled={isPending}
                                        width='100%'
                                    />
                                ))}
                            </>
                        )}
                        {step === 'connect' && (
                            <AwsCredentials
                                profileName={profileName}
                                onProfileChange={(value) => {
                                    setProfileName(value);
                                    resolve.reset();
                                    create.reset();
                                    update.reset();
                                }}
                                isDisabled={isPending}
                            />
                        )}
                        {error && <IssueErrorAlert error={error} />}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        Cancel
                    </Button>
                    {step === 'connect' && (
                        <Button
                            variant='secondary'
                            type='button'
                            isDisabled={isPending}
                            onPress={() => {
                                setStep('configure');
                                resolve.reset();
                                create.reset();
                                update.reset();
                            }}
                        >
                            Back
                        </Button>
                    )}
                    <Button
                        variant='accent'
                        type='submit'
                        isDisabled={!isConfigurationValid || (step === 'connect' && !profileName.trim()) || isPending}
                        isPending={isPending}
                    >
                        {step === 'configure'
                            ? 'Next'
                            : isPending
                              ? 'Connecting…'
                              : remoteTrainer
                                ? 'Save changes'
                                : 'Connect'}
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
