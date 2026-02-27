import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Form,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    TextField,
} from '@geti/ui';

import { $api } from '../../api/client';
import { SchemaModel, SchemaTrainJobPayload } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { SchemaTrainJob } from './train-model';

export const RetrainModelModal = ({
    baseModel,
    close,
}: {
    baseModel: SchemaModel;
    close: (job: SchemaTrainJob | undefined) => void;
}) => {
    const { datasets, id: project_id } = useProject();
    const [name, setName] = useState<string>(baseModel.name);
    const [selectedDataset, setSelectedDataset] = useState<Key | null>(baseModel.dataset_id);
    const [maxSteps, setMaxSteps] = useState<number>(10000);

    const trainMutation = $api.useMutation('post', '/api/jobs:train');

    const save = () => {
        const dataset_id = selectedDataset?.toString();

        if (!dataset_id) {
            return;
        }

        const payload: SchemaTrainJobPayload = {
            dataset_id,
            project_id,
            model_name: name,
            policy: baseModel.policy,
            max_steps: maxSteps,
            base_model_id: baseModel.id!,
        };
        trainMutation.mutateAsync({ body: payload }).then((response) => {
            close(response as SchemaTrainJob | undefined);
        });
    };

    return (
        <Dialog>
            <Heading>Retrain Model</Heading>
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
                    <Picker label='Dataset' selectedKey={selectedDataset} onSelectionChange={setSelectedDataset}>
                        {datasets.map((dataset) => (
                            <Item key={dataset.id}>{dataset.name}</Item>
                        ))}
                    </Picker>
                    <TextField label='Policy' value={baseModel.policy.toUpperCase()} isReadOnly />
                    <NumberField
                        label='Max Steps'
                        value={maxSteps}
                        onChange={setMaxSteps}
                        minValue={100}
                        maxValue={100000}
                        step={100}
                    />
                </Form>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => close(undefined)}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={save}>
                    Retrain
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
