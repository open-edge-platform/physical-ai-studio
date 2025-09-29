import { useState } from 'react';

import { Button, ButtonGroup, Flex, Form, Heading, TextField, View } from '@geti/ui';
import { useNavigate } from 'react-router';

import { $api } from '../../api/client';
import { SchemaProjectInput } from '../../api/openapi-spec';
import { paths } from '../../router';

export const NewProjectPage = () => {
    const navigate = useNavigate();
    const saveMutation = $api.useMutation('post', '/api/projects');
    const [project, setProject] = useState<SchemaProjectInput>({
        name: '',
        datasets: [],
    });

    const isValid = () => {
        return project.name !== '';
    };

    const save = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        saveMutation
            .mutateAsync({
                body: project,
            })
            .then(({ id }) => {
                navigate(paths.project.datasets.index({ project_id: id! }));
            });
    };

    return (
        <Flex width='100%' height='100%' direction={'column'} alignItems='center'>
            <View flex={1} width={'100%'} maxWidth='1320px' paddingTop='size-400'>
                <Form maxWidth='size-3600' onSubmit={save}>
                    <Flex justifyContent={'space-between'}>
                        <Heading>New Project</Heading>
                        <ButtonGroup>
                            <Button variant='secondary' href={paths.projects.index.pattern}>
                                Cancel
                            </Button>
                            <Button isDisabled={!isValid()} type='submit'>
                                Save
                            </Button>
                        </ButtonGroup>
                    </Flex>
                    <TextField
                        isRequired
                        label='name'
                        value={project.name}
                        onChange={(name) => setProject((p) => ({ ...p, name }))}
                    />
                </Form>
            </View>
        </Flex>
    );
};
