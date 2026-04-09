import { useState, type FormEvent } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, TextField } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';

type Dataset = SchemaDatasetOutput;

export const RenameDatasetDialog = ({
    dataset,
    onDone,
}: {
    dataset: Dataset;
    onDone: (dataset: SchemaDatasetOutput | undefined) => void;
}) => {
    const [name, setName] = useState(dataset.name);
    const renameMutation = $api.useMutation('put', '/api/dataset/{dataset_id}');

    const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const trimmedName = name.trim();
        if (trimmedName.length === 0 || dataset.id === undefined) {
            return;
        }

        const updated = await renameMutation.mutateAsync({
            params: {
                path: {
                    dataset_id: dataset.id,
                },
            },
            body: {
                name: trimmedName,
            },
        });

        onDone(updated);
    };

    return (
        <form onSubmit={onSubmit}>
            <Dialog width='size-6000'>
                <Heading>Rename dataset</Heading>
                <Divider />
                <Content>
                    <TextField
                        // eslint-disable-next-line jsx-a11y/no-autofocus
                        autoFocus
                        isRequired
                        width='100%'
                        label='Dataset name'
                        value={name}
                        onChange={setName}
                    />
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={() => onDone(undefined)}>
                        Cancel
                    </Button>
                    <Button
                        variant='accent'
                        type='submit'
                        isDisabled={name.trim() === ''}
                        isPending={renameMutation.isPending}
                    >
                        Save
                    </Button>
                </ButtonGroup>
            </Dialog>
        </form>
    );
};
