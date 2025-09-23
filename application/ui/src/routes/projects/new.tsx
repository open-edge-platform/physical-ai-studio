import { Button, ButtonGroup, Flex, Heading, Form, NumberField, TextField, Item, TabList, TabPanels, Tabs, Text, View } from '@geti/ui';
import { Contract, FitScreen, Gear } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { $api } from '../../api/client';
import { useState } from 'react';
import { SchemaProject } from '../../api/openapi-spec';
import { paths } from '../../router';

export const NewProjectPage = () => {
    const navigate = useNavigate();
    const saveMutation = $api.useMutation('put', '/api/projects');
    const [project, setProject] = useState<SchemaProject>({
        name: ""
    });

    const isValid = () => {
        return project.name !== "";
    }

    const save = () => {
        saveMutation
            .mutateAsync({
                body: project,
            })
            .then((project) => {
                navigate(paths.project.datasets.index({ project_id: project.id }));
            });
    };

    return (
        <Flex width='100%' height='100%' direction={'column'} alignItems='center'>
            <View flex={1} width={'100%'} maxWidth='1320px' paddingTop='size-400'>
                <Flex justifyContent={'space-between'}>
                    <Heading>New Project</Heading>
                    <ButtonGroup>
                        <Button variant='secondary' href={paths.projects.index.pattern}>
                            Cancel
                        </Button>
                        <Button isDisabled={!isValid() || saveMutation.isPending} onPress={save}>
                            Save
                        </Button>
                    </ButtonGroup>
                </Flex>
                <Form maxWidth='size-3600'>
                    <TextField label='name' value={project.name} onChange={(name) => setProject((p) => ({ ...p, name }))} />
                </Form>
            </View>
        </Flex>
    );
};
