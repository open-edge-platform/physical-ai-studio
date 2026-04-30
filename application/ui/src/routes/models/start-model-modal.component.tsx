import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { SchemaModel } from '../../api/openapi-spec';
import { $api } from '../../api/client';
import { BackendSelection, defaultBackend } from '../../features/configuration/shared/backend-selection';
import { paths } from '../../router';

const backendLabels: Record<string, string> = {
    torch: 'Torch',
    openvino: 'OpenVINO',
    onnx: 'ONNX',
    executorch: 'ExecuTorch',
};

interface StartInferenceDialogProps {
    close: () => void;
    model: SchemaModel;
}

export const StartInferenceDialog = ({ close, model }: StartInferenceDialogProps) => {
    const { data: policyBackends } = $api.useQuery('get', '/api/policies/backends');

    const availableBackends = (() => {
        const policySupported = policyBackends?.[model.policy] ?? [];
        const modelExported = model.available_backends;
        const intersection = policySupported.filter((b) => modelExported.includes(b));
        return intersection.map((id) => ({ id, name: backendLabels[id] ?? id }));
    })();

    const [backend, setBackend] = useState<string>(availableBackends[0]?.id ?? defaultBackend);

    const navigate = useNavigate();
    const onStart = () => {
        close();
        navigate(
            paths.project.models.inference({
                project_id: model.project_id,
                model_id: model.id!,
                backend,
            })
        );
    };

    return (
        <Dialog>
            <Heading>Run model</Heading>
            <Divider />
            <Content>
                <BackendSelection backend={backend} setBackend={setBackend} backends={availableBackends} />
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={onStart} isDisabled={availableBackends.length === 0}>
                    Start
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
