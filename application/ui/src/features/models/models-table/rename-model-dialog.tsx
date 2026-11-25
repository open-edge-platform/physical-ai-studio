import { useState, type FormEvent } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, TextField } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaModel } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';

export const RenameModelDialog = ({ model, close }: { model: SchemaModel; close: () => void }) => {
    const [name, setName] = useState(model.name);
    const renameMutation = $api.useMutation('patch', '/api/models/{model_id}', {
        meta: {
            invalidates: [
                ['get', '/api/models/{model_id}', { params: { path: { model_id: model.id! } } }],
                ['get', '/api/projects/{project_id}/models', { params: { path: { project_id: model.project_id } } }],
            ],
        },
        onSuccess: close,
    });
    const isSaveDisabled = !name.trim() || name.trim() === model.name || renameMutation.isPending;

    const onSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (isSaveDisabled || model.id === undefined) {
            return;
        }
        renameMutation.mutate({
            params: { path: { model_id: model.id } },
            body: { name: name.trim() },
        });
    };

    return (
        <form onSubmit={onSubmit}>
            <Dialog width='size-6000'>
                <Heading>Rename model</Heading>
                <Divider />
                <Content>
                    <TextField
                        // eslint-disable-next-line jsx-a11y/no-autofocus
                        autoFocus
                        isRequired
                        width='100%'
                        label='Model name'
                        value={name}
                        onChange={setName}
                    />
                    {renameMutation.isError && (
                        <InlineAlert variant='error'>
                            {getApiErrorMessage(renameMutation.error) ?? 'The model could not be renamed. Try again.'}
                        </InlineAlert>
                    )}
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={close} isDisabled={renameMutation.isPending}>
                        Cancel
                    </Button>
                    <Button
                        variant='accent'
                        type='submit'
                        isDisabled={isSaveDisabled}
                        isPending={renameMutation.isPending}
                    >
                        Save
                    </Button>
                </ButtonGroup>
            </Dialog>
        </form>
    );
};
