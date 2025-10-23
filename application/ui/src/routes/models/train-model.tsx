import { Dialog, Heading, Form, Header, Key, Picker, Item, Divider, Content, Text, ButtonGroup, Button, TextField } from "@geti/ui"
import { $api } from "../../api/client"
import { useProject } from "../../features/projects/use-project"
import { v4 as uuidv4 } from 'uuid';
import { useState } from "react";
import { SchemaModel } from "../../api/openapi-spec";

export const TrainModelModal = (close: () => void) => {
    const { datasets } = useProject();
    const [name, setName] = useState<string>("");
    const [selectedDatasets, setSelectedDatasets] = useState<Key | null>(null);
    const [selectedPolicy, setSelectedPolicy] = useState<Key | null>("act");

    const save = () => {
        console.log({
            id: uuidv4(),
            name,
            path: "",
            properties: {
                "dataset": selectedDatasets?.toString() ?? "",
                "policy": selectedPolicy?.toString() ?? "",
            }
            
        });
        //saveMutation.mutateAsync(
        //    { body: { id, name, datasets: [] } },
        //    {
        //        onSuccess: () => {
        //            navigate(paths.project.datasets.index({ project_id: id }));
        //        },
        //    }
        //);
    };

    return (
        <Dialog>
            <Heading>Train Model</Heading>
            <Divider />
            <Content>
                <Form onSubmit={(e) => { e.preventDefault(); save()}} validationBehavior="native">
                    <TextField label="Name" value={name} onChange={setName} />
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