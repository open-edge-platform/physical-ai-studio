import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Form, Heading, Item, Key, Picker, TextField } from '@geti/ui';

import { $api } from '../../api/client';
import { SchemaJob, SchemaTrainJobPayload } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';

export type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaTrainJobPayload;
};

export const TrainModelModal = (close: (job: SchemaTrainJob | undefined) => void) => {
    const { datasets, id: project_id } = useProject();
    const [name, setName] = useState<string>('');
    const [selectedDatasets, setSelectedDatasets] = useState<Key | null>(null);
    const [selectedPolicy, setSelectedPolicy] = useState<Key | null>('act');

    const trainMutation = $api.useMutation('post', '/api/jobs/train');

    const save = () => {
        const dataset_id = selectedDatasets?.toString();
        const policy = selectedPolicy?.toString();

        if (!dataset_id || !policy) {
            return;
        }

        const payload: SchemaTrainJobPayload = {
            dataset_id: dataset_id!,
            project_id,
            model_name: name,
            policy: policy!,
        };
        trainMutation.mutateAsync({ body: payload }).then((response) => {
            close(response as SchemaTrainJob | undefined);
        });
    };

    return (
        <Dialog>
            <Heading>Train Model</Heading>
            <Divider />
            <Content>
                <Form
                    onSubmit={(e) => {
                        e.preventDefault();
                        save();
                    }}
                    validationBehavior='native'
                >
                    <TextField label='Name' value={name} onChange={setName} />
                    <Picker label='Dataset' selectedKey={selectedDatasets} onSelectionChange={setSelectedDatasets}>
                        {datasets.map((dataset) => (
                            <Item key={dataset.id}>{dataset.name}</Item>
                        ))}
                    </Picker>
                    <Picker label='Policy' selectedKey={selectedPolicy} onSelectionChange={setSelectedPolicy}>
                        <Item key='act'>Act</Item>
                    </Picker>
                </Form>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={save}>
                    Train
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
