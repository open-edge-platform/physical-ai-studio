import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, DialogTrigger, Divider, Form, Heading, TextField } from '@geti/ui';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../api/client';
import { useSettings } from '../../components/settings/use-settings';
import { makeNameSafeForPath } from './record/utils';

interface NewDatasetFormProps {
    project_id: string;
    onDone: () => void;
}
const NewDatasetForm = ({ project_id, onDone }: NewDatasetFormProps) => {
    const saveMutation = $api.useMutation('post', '/api/dataset');
    const [name, setName] = useState<string>('');
    const { geti_action_dataset_path } = useSettings();

    const save = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const id = uuidv4();
        await saveMutation.mutateAsync({
            body: {
                id,
                name,
                project_id,
                path: `${geti_action_dataset_path}/${makeNameSafeForPath(name)}`,
            },
        });
        onDone()
    };


    return (
        <Form onSubmit={save} width={'size-6000'} validationBehavior='native'>
            <Dialog>
                <Heading>Create dataset</Heading>
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
                    <Button variant='secondary' onPress={onDone}>
                        Cancel
                    </Button>
                    <Button
                        variant='accent'
                        type='submit'
                        isDisabled={name === ''}
                        isPending={saveMutation.isPending}
                    >
                        Save
                    </Button>
                </ButtonGroup>
            </Dialog>
        </Form>
    )

}
export const NewDatasetLink = ({ project_id }: { project_id: string }) => {
    return (
        <DialogTrigger>
            <Button variant='accent'>New Dataset</Button>
            {(close) => (
                <NewDatasetForm project_id={project_id} onDone={close} />
            )}
        </DialogTrigger>
    );
};

interface NewDatasetDialogContainerProps {
    project_id: string;
    show: boolean;
    onDismiss: () => void;
}
export const NewDatasetDialogContainer = ({ project_id, show, onDismiss }: NewDatasetDialogContainerProps) => {
    return (
        <DialogContainer onDismiss={onDismiss}>
            {show && (
                <NewDatasetForm project_id={project_id} onDone={onDismiss} />
            )}
        </DialogContainer>
    );
};
