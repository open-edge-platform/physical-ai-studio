import { FormEvent, ReactElement, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogTrigger, Divider, Form, Heading, TextField } from '@geti-ui/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../../api/client';
import { paths } from '../../../../router';

type CreateProjectProps = {
    trigger: ReactElement;
};

export const CreateProject = ({ trigger }: CreateProjectProps) => {
    const navigate = useNavigate();
    const saveMutation = $api.useMutation('post', '/api/projects', {
        meta: {
            invalidates: [['get', '/api/projects']],
        },
    });
    const [name, setName] = useState<string>('');

    const save = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const id = uuidv4();
        saveMutation.mutate(
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
            {trigger}
            {(close) => (
                <Dialog width={'size-6000'}>
                    <Heading>Create project</Heading>
                    <Divider />
                    <Content>
                        <Form id={'create-project-form'} onSubmit={save} validationBehavior='native'>
                            <TextField
                                // eslint-disable-next-line jsx-a11y/no-autofocus
                                autoFocus
                                isRequired
                                width='100%'
                                label='Project name'
                                value={name}
                                onChange={setName}
                            />
                        </Form>
                    </Content>
                    <ButtonGroup>
                        <Button variant='secondary' onPress={close}>
                            Cancel
                        </Button>
                        <Button
                            form={'create-project-form'}
                            variant='accent'
                            type='submit'
                            isDisabled={name === ''}
                            isPending={saveMutation.isPending}
                        >
                            Save
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
