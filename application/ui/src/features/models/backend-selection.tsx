import { Item, Picker } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { SchemaModel } from '../../api/openapi-spec';

export const defaultBackend = 'torch';

const backendLabels: Record<string, string> = {
    torch: 'Torch',
    openvino: 'OpenVINO',
    onnx: 'ONNX',
    executorch: 'ExecuTorch',
};

interface BackendSelectionProps {
    model: SchemaModel;
    backend: string;
    setBackend: (backend: string) => void;
}

export const BackendSelection = ({ model, backend, setBackend }: BackendSelectionProps) => {
    const { data: policyBackends } = $api.useSuspenseQuery('get', '/api/policies/backends');

    const items = (policyBackends[model.policy] ?? [])
        .filter((b) => model.available_backends.includes(b))
        .map((id) => ({ id, name: backendLabels[id] ?? id }));

    return (
        <Picker
            items={items}
            selectedKey={backend}
            label='Backend'
            onSelectionChange={(key) => setBackend(key!.toString())}
            flex={1}
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
