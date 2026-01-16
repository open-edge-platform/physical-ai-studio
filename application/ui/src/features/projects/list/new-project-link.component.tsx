import { useState } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Form,
    Heading,
    TextField,
} from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { paths } from '../../../router';

import classes from './project-list.module.scss';

export const NewProjectLink = ({ className }: { className?: string }) => {
    const navigate = useNavigate();
    const saveMutation = $api.useMutation('post', '/api/projects');
    const [name, setName] = useState<string>('');

    const save = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const id = uuidv4();
        saveMutation.mutateAsync(
            { body: { id, name, datasets: [] } },
            {
                onSuccess: () => {
                    navigate(paths.project.robots.new({ project_id: id }));
                },
            }
        );
    };

    return (
        <DialogTrigger>
            <ActionButton UNSAFE_className={className ?? classes.link} height={'100%'}>
                <AddCircle />
                Add project
            </ActionButton>
            {(close) => (
                <Form onSubmit={save} width={'size-6000'} validationBehavior='native'>
                    <Dialog>
                        <Heading>Add project</Heading>
                        <Divider />
                        <Content>
                            <TextField
                                // eslint-disable-next-line jsx-a11y/no-autofocus
                                autoFocus
                                isRequired
                                width='100%'
                                label='Project name'
                                value={name}
                                onChange={setName}
                            />
                        </Content>
                        <ButtonGroup>
                            <Button variant='secondary' onPress={close}>
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
            )}
        </DialogTrigger>
    );
};
