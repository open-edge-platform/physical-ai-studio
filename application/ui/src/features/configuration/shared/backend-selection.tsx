import { Item, Picker } from '@geti/ui';

interface BackendSelectionProps {
    backend: string;
    setBackend: (backend: string) => void;
}

export const availableBackends = [
    { id: 'torch', name: 'Torch' },
    //{ id: 'openvino', name: 'OpenVINO' },
    //{ id: 'onnx', name: 'ONNX' },
    //{ id: 'torch_export_ir', name: 'Torch Export IR' },
];

export const BackendSelection = ({ backend, setBackend }: BackendSelectionProps) => {
    return (
        <Picker
            items={availableBackends}
            selectedKey={backend}
            label='Backend'
            onSelectionChange={(m) => setBackend(m!.toString())}
            flex={1}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
