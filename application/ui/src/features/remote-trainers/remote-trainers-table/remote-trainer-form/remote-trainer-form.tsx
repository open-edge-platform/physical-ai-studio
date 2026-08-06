import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Form, Heading, Text, TextField } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { getApiErrorMessage } from '../../../../api/errors';
import { SchemaRemoteTrainer } from '../../../../api/openapi-spec';

import classes from './remote-trainer-form.module.css';

type RemoteTrainerFormProps = {
    remoteTrainer?: SchemaRemoteTrainer;
    close: () => void;
};

export const RemoteTrainerForm = ({ remoteTrainer, close }: RemoteTrainerFormProps) => {
    const [name, setName] = useState(remoteTrainer?.name ?? '');
    const [url, setUrl] = useState(remoteTrainer?.url ?? '');
    const [error, setError] = useState<string>();
    const isEditing = remoteTrainer !== undefined;

    const createMutation = $api.useMutation('post', '/api/remote-trainers', {
        meta: {
            invalidates: [['get', '/api/remote-trainers']],
        },
    });
    const updateMutation = $api.useMutation('patch', '/api/remote-trainers/{remote_trainer_id}', {
        meta: {
            invalidates: [['get', '/api/remote-trainers']],
        },
    });
    const isPending = createMutation.isPending || updateMutation.isPending;

    const save = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError(undefined);
        const normalizedName = name.trim();

        try {
            if (isEditing) {
                await updateMutation.mutateAsync({
                    params: { path: { remote_trainer_id: remoteTrainer.id } },
                    body: { name: normalizedName, url },
                });
            } else {
                await createMutation.mutateAsync({ body: { name: normalizedName, url } });
            }
            close();
        } catch (mutationError) {
            setError(getApiErrorMessage(mutationError) ?? 'The remote trainer could not be saved. Try again.');
        }
    };

    return (
        <Form onSubmit={save} validationBehavior='native' width='size-6000'>
            <Dialog>
                <Heading>{isEditing ? 'Edit remote trainer' : 'Add remote trainer'}</Heading>
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
                        <TextField
                            isRequired
                            label='Trainer URL'
                            type='url'
                            value={url}
                            onChange={setUrl}
                            description='Use the endpoint URL that accepts Physical AI Studio training jobs.'
                            width='100%'
                        />
                        {error && <Text UNSAFE_className={classes.errorMessage}>{error}</Text>}
                    </Flex>
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={isPending}>
                        Cancel
                    </Button>
                    <Button variant='accent' type='submit' isDisabled={!name.trim() || !url} isPending={isPending}>
                        {isEditing ? 'Save changes' : 'Add trainer'}
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
