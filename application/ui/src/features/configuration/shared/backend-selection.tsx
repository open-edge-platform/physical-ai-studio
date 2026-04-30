import { Item, Picker } from '@geti-ui/ui';

interface BackendItem {
    id: string;
    name: string;
}

interface BackendSelectionProps {
    backend: string;
    setBackend: (backend: string) => void;
    backends?: BackendItem[];
}

const backendLabels: Record<string, string> = {
    torch: 'Torch',
    openvino: 'OpenVINO',
    onnx: 'ONNX',
    executorch: 'ExecuTorch',
};

export const defaultBackend = 'torch';

export const BackendSelection = ({ backend, setBackend, backends }: BackendSelectionProps) => {
    const items = backends ?? [{ id: defaultBackend, name: backendLabels[defaultBackend] }];

    return (
        <Picker
            items={items}
            selectedKey={backend}
            label='Backend'
            onSelectionChange={(m) => setBackend(m!.toString())}
            flex={1}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
