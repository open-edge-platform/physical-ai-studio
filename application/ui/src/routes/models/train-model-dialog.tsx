import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Flex,
    Form,
    Heading,
    Item,
    Key,
    NumberField,
    Picker,
    TextField,
} from '@geti/ui';

import { $api } from '../../api/client';
import { SchemaJob, SchemaModel, SchemaTrainJobPayload } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';

export type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaTrainJobPayload;
};

interface TrainModelDialogProps {
    baseModel?: SchemaModel;
    close: (job: SchemaTrainJob | undefined) => void;
    defaultMaxSteps?: number;
}

export const TrainModelDialog = ({ baseModel, close, defaultMaxSteps = 10000 }: TrainModelDialogProps) => {
    const defaultName = baseModel?.name ?? '';
    const defaultDatasetId = baseModel?.dataset_id ?? null;
    const extraPayload = baseModel ? { base_model_id: baseModel.id! } : undefined;

    const [selectedPolicy, setSelectedPolicy] = useState<Key | null>(baseModel?.policy ?? 'act');
    const { datasets, id: projectId } = useProject();

    const [name, setName] = useState<string>(defaultName);
    const [selectedDataset, setSelectedDataset] = useState<Key | null>(defaultDatasetId);
    const [maxSteps, setMaxSteps] = useState<number>(defaultMaxSteps);
    const [batchSize, setBatchSize] = useState<number>(8);

    const trainMutation = $api.useMutation('post', '/api/jobs:train');

    const save = () => {
        const dataset_id = selectedDataset?.toString();

        if (!dataset_id || !selectedPolicy) {
            return;
        }

        const payload: SchemaTrainJobPayload = {
            dataset_id,
            project_id: projectId,
            model_name: name,
            policy: selectedPolicy.toString(),
            max_steps: maxSteps,
            batch_size: batchSize,
            ...extraPayload,
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
                    <Picker label='Dataset' selectedKey={selectedDataset} onSelectionChange={setSelectedDataset}>
                        {datasets.map((dataset) => (
                            <Item key={dataset.id}>{dataset.name}</Item>
                        ))}
                    </Picker>
                    <Picker
                        label='Policy'
                        selectedKey={selectedPolicy}
                        onSelectionChange={setSelectedPolicy}
                        isDisabled={baseModel !== undefined}
                    >
                        <Item key='act'>Act</Item>
                        <Item key='pi0'>Pi0</Item>
                        <Item key='smolvla'>SmolVLA</Item>
                    </Picker>
                    <Disclosure isQuiet UNSAFE_style={{ padding: 0 }}>
                        <DisclosureTitle UNSAFE_style={{ fontSize: 13, padding: '4px 0' }}>
                            Advanced settings
                        </DisclosureTitle>
                        <DisclosurePanel UNSAFE_style={{ padding: 0 }}>
                            <Flex direction='row' gap='size-150' width='100%'>
                                <NumberField
                                    label='Max Steps'
                                    value={maxSteps}
                                    onChange={setMaxSteps}
                                    minValue={100}
                                    maxValue={100000}
                                    step={100}
                                    flex
                                />
                                <NumberField
                                    label='Batch Size'
                                    value={batchSize}
                                    onChange={setBatchSize}
                                    minValue={1}
                                    maxValue={256}
                                    step={1}
                                    flex
                                />
                            </Flex>
                        </DisclosurePanel>
                    </Disclosure>
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
