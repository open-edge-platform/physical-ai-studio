import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogTrigger, Divider, Form, Heading, TextField } from '@geti-ui/ui';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';

interface NewDatasetFormProps {
    project_id: string;
    onDone: (dataset: SchemaDatasetOutput | undefined) => void;
}
export const NewDatasetForm = ({ project_id, onDone }: NewDatasetFormProps) => {
    const saveMutation = $api.useMutation('post', '/api/dataset', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}', { params: { path: { project_id } } }]],
        },
    });
    const [name, setName] = useState<string>('');
    const [defaultTask, setDefaultTask] = useState<string>('');

    const save = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const id = uuidv4();
        const result = await saveMutation.mutateAsync({
            body: {
                id,
                name,
                project_id,
                default_task: defaultTask,
            },
        });
        onDone(result);
    };

    return (
        <Form onSubmit={save} width={'size-6000'} validationBehavior='native'>
            <Dialog>
                <Heading>Create new dataset</Heading>
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

                    <TextField width='100%' label='Task' value={defaultTask} onChange={setDefaultTask} />
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={() => onDone(undefined)}>
                        Cancel
                    </Button>
                    <Button
                        variant='accent'
                        type='submit'
                        isDisabled={name.trim() === ''}
                        isPending={saveMutation.isPending}
                    >
                        Save
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    );
};
export const NewDatasetLink = ({ project_id }: { project_id: string }) => {
    return (
        <DialogTrigger>
            <Button variant='accent'>Create new dataset</Button>
            {(close) => <NewDatasetForm project_id={project_id} onDone={close} />}
        </DialogTrigger>
    );
};
