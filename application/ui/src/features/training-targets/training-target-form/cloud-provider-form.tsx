import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Form, Heading, Link, Text, TextField } from '@geti-ui/ui';
import { ExternalLinkIcon } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaRemoteTrainer } from '../../../api/openapi-spec';
import { ReactComponent as AwsIcon } from '../../../assets/icons/aws-icon.svg';

import classes from '../training-targets-table/remote-trainer-form/remote-trainer-form.module.css';

const AWS_STACK_URL =
    'https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1' +
    '#/stacks/create/review?templateURL=' +
    encodeURIComponent(
        'https://physical-ai-studio.s3.eu-west-1.amazonaws.com/aws-cf-templates/aws-batch-trainer.yaml'
    ) +
    '&stackName=physical-ai-studio-batch-trainer';

interface CloudProviderFormProps {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
}

export const CloudProviderForm = ({ remoteTrainer, close }: CloudProviderFormProps) => {
    const existing = remoteTrainer?.connection.connection_mode === 'aws_batch' ? remoteTrainer.connection : undefined;
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
    const errorMessage =
        error && typeof error === 'object' && 'detail' in error && typeof error.detail === 'string'
            ? error.detail
            : getApiErrorMessage(error);
    const canSubmit =
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
        if (!canSubmit || isPending) return;
        create.reset();
        update.reset();
        try {
            const connection = await resolve.mutateAsync({
                body: {
                    configuration_uri: values.configuration_uri.trim(),
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
            <Dialog width='size-6000'>
                <Heading>{remoteTrainer ? 'Edit training target' : 'Add training target'}</Heading>
                <Divider />
                <Content>
                    <Flex direction='column' gap='size-150'>
                        <Heading level={3}>AWS Provider</Heading>
                        {isLoadingSchema && <Text>Loading configuration fields...</Text>}
                        {(schemaError || (!isLoadingSchema && !provider)) && (
                            <div role='alert'>AWS configuration fields could not be loaded.</div>
                        )}
                        {provider?.fields.map((field) => (
                            <TextField
                                key={field.name}
                                label={field.title}
                                isRequired
                                value={values[field.name] ?? ''}
                                onChange={(value) => setValues((current) => ({ ...current, [field.name]: value }))}
                                minLength={field.minLength ?? undefined}
                                maxLength={field.maxLength ?? undefined}
                                pattern={field.pattern ?? undefined}
                                isDisabled={isPending}
                                width='100%'
                            />
                        ))}
                        {error && <div role='alert'>{errorMessage ?? 'The AWS provider could not be saved.'}</div>}
                        {!remoteTrainer && (
                            <>
                                <Divider size='S' />
                                <Flex direction='column' alignItems='start' gap='size-50'>
                                    <AwsIcon aria-hidden='true' className={classes.awsIcon} />
                                    <Link href={AWS_STACK_URL} target='_blank' rel='noopener noreferrer'>
                                        <span className={classes.awsStackLinkContent}>
                                            Deploy AWS Batch stack
                                            <ExternalLinkIcon aria-hidden='true' />
                                        </span>
                                    </Link>
                                </Flex>
                            </>
                        )}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        Cancel
                    </Button>
                    <Button variant='accent' type='submit' isDisabled={!canSubmit || isPending} isPending={isPending}>
                        {remoteTrainer ? 'Save changes' : 'Add training target'}
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
