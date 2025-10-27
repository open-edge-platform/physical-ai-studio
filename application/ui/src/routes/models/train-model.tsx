import { Dialog, Heading, Form, Key, Picker, Item, Divider, Content, ButtonGroup, Button, TextField } from "@geti/ui"
import { $api } from "../../api/client"
import { useProject } from "../../features/projects/use-project"
import { useState } from "react";
import { SchemaTrainJobPayload } from "../../api/openapi-spec";

export const TrainModelModal = (close: () => void) => {
    const { datasets, id: project_id } = useProject();
    const [name, setName] = useState<string>("");
    const [selectedDatasets, setSelectedDatasets] = useState<Key | null>(null);
    const [selectedPolicy, setSelectedPolicy] = useState<Key | null>("act");

    const trainMutation = $api.useMutation('post','/api/jobs/train')

    const save = () => {
        const dataset_id = selectedDatasets?.toString()
        const policy = selectedPolicy?.toString()!

        if (!dataset_id || !policy) {
            return;
        }

        const payload: SchemaTrainJobPayload = {
            dataset_id: dataset_id!,
            project_id,
            model_name: name,
            policy: policy!

        }
        trainMutation.mutate({ body: payload })
    };

    return (
        <Dialog>
            <Heading>Train Model</Heading>
            <Divider />
            <Content>
                <Form onSubmit={(e) => { e.preventDefault(); save()}} validationBehavior="native">
                    <TextField label="Name" autoFocus value={name} onChange={setName} />
                    <Picker label="Dataset" selectedKey={selectedDatasets} onSelectionChange={setSelectedDatasets}>
                        {datasets.map((dataset) => <Item key={dataset.id}>{dataset.name}</Item>)}
                    </Picker>
                    <Picker label="Policy" selectedKey={selectedPolicy} onSelectionChange={setSelectedPolicy}>
                        <Item key="act">Act</Item>
                    </Picker>
                </Form>
            </Content>
            <ButtonGroup>
                <Button variant="secondary" onPress={close}>Cancel</Button>
                <Button variant="accent" onPress={save}>Train</Button>
            </ButtonGroup>
        </Dialog>
    )
}
