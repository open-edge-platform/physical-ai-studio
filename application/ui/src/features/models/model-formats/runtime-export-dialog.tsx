import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, Item, Picker } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaModel } from '../../../api/openapi-spec';
import { getDefaultInferenceDevice, getSupportedInferenceDevices } from '../backend-selection/backend-selection';
import { runtimeExportUrl } from '../runtime-export';

interface RuntimeExportDialogProps {
    close: () => void;
    model: SchemaModel;
    backend: string;
}

export const RuntimeExportDialog = ({ close, model, backend }: RuntimeExportDialogProps) => {
    const { data: environments = [] } = $api.useQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id: model.project_id } },
    });
    const { data: dataset } = $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        { params: { path: { dataset_id: model.dataset_id! } } },
        { enabled: model.dataset_id != null }
    );
    const { data: tasks = [] } = $api.useQuery(
        'get',
        '/api/models/{model_id}/tasks',
        { params: { path: { model_id: model.id! } } },
        { enabled: model.dataset_id != null }
    );
    const { data: inferenceDevices = [] } = $api.useQuery('get', '/api/system/devices/inference');

    const defaultEnvironmentId = dataset?.environment_id ?? environments[0]?.id;
    const [environmentId, setEnvironmentId] = useState<string | undefined>(defaultEnvironmentId);
    const selectedEnvironmentId = environmentId ?? defaultEnvironmentId;

    const devices = getSupportedInferenceDevices(inferenceDevices, backend);
    const defaultDevice = getDefaultInferenceDevice(inferenceDevices, backend);
    const [device, setDevice] = useState<string | undefined>(defaultDevice?.device);
    const selectedDevice = device ?? defaultDevice?.device;

    const [task, setTask] = useState<string | undefined>(tasks[0]);
    const selectedTask = task ?? tasks[0];

    const downloadUrl =
        model.id !== undefined && selectedEnvironmentId !== undefined && selectedDevice !== undefined
            ? runtimeExportUrl({
                  modelId: model.id,
                  environmentId: selectedEnvironmentId,
                  backend,
                  device: selectedDevice,
                  task: selectedTask,
              })
            : undefined;

    return (
        <Dialog>
            <Heading>Download runtime export</Heading>
            <Divider />
            <Content>
                <Picker
                    isRequired
                    label='Environment'
                    items={environments}
                    selectedKey={selectedEnvironmentId}
                    onSelectionChange={(key) => setEnvironmentId(key === null ? undefined : String(key))}
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>
                <Picker
                    isRequired
                    label='Device'
                    items={devices}
                    selectedKey={selectedDevice}
                    onSelectionChange={(key) => setDevice(key === null ? undefined : String(key))}
                >
                    {(item) => <Item key={item.device}>{item.name}</Item>}
                </Picker>
                {tasks.length > 0 && (
                    <Picker
                        label='Task'
                        items={tasks.map((value) => ({ id: value, name: value }))}
                        selectedKey={selectedTask}
                        onSelectionChange={(key) => setTask(key === null ? undefined : String(key))}
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>
                )}
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    href={downloadUrl}
                    isDisabled={downloadUrl === undefined}
                    onPress={close}
                    target='_blank'
                    rel='noopener noreferrer'
                >
                    Download
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
