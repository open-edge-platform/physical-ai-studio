import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    DialogTrigger,
    Divider,
    Form,
    Heading,
    Item,
    Picker,
    TextField,
} from '@geti/ui';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { useSettings } from '../../components/settings/use-settings';
import { makeNameSafeForPath } from './record/utils';

interface NewDatasetFormProps {
    project_id: string;
    onDone: (dataset: SchemaDatasetOutput | undefined) => void;
}
const NewDatasetForm = ({ project_id, onDone }: NewDatasetFormProps) => {
    const saveMutation = $api.useMutation('post', '/api/dataset');
    const [name, setName] = useState<string>('');
    const { geti_action_dataset_path } = useSettings();

    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: {
            path: {
                project_id,
            },
        },
    });

    const [environmentId, setEnvironmentId] = useState<string | undefined>(environments[0]?.id);

    const save = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        if (environmentId !== undefined) {
            const id = uuidv4();
            const result = await saveMutation.mutateAsync({
                body: {
                    id,
                    name,
                    project_id,
                    environment_id: environmentId,
                    path: `${geti_action_dataset_path}/${makeNameSafeForPath(name)}`,
                },
            });
            onDone(result);
        }
    };

    return (
        <Form onSubmit={save} width={'size-6000'} validationBehavior='native'>
            <Dialog>
                <Heading>Create dataset</Heading>
                <Divider />
                <Content>
                    <Picker
                        items={environments}
                        selectedKey={environmentId}
                        label='Environment'
                        onSelectionChange={(m) => setEnvironmentId(m === null ? undefined : m.toString())}
                        flex={1}
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>

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
                    <Button variant='accent' type='submit' isDisabled={name === ''} isPending={saveMutation.isPending}>
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
            <Button variant='accent'>New Dataset</Button>
            {(close) => <NewDatasetForm project_id={project_id} onDone={close} />}
        </DialogTrigger>
    );
};

interface NewDatasetDialogContainerProps {
    project_id: string;
    show: boolean;
    onDismiss: (dataset: SchemaDatasetOutput | undefined) => void;
}
export const NewDatasetDialogContainer = ({ project_id, show, onDismiss }: NewDatasetDialogContainerProps) => {
    return (
        <DialogContainer onDismiss={() => onDismiss(undefined)}>
            {show && <NewDatasetForm project_id={project_id} onDone={onDismiss} />}
        </DialogContainer>
    );
};
