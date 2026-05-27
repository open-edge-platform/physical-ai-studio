import { Suspense, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading, ProgressCircle } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { SchemaModel } from '../../api/openapi-spec';
import { BackendSelection, defaultBackend } from '../../features/models/backend-selection';
import { paths } from '../../router';

const getDefaultbackend = (model: SchemaModel) => {
    if (model.available_backends.includes(defaultBackend)) {
        return defaultBackend;
    }

    return model.available_backends.at(0) ?? defaultBackend;
};

interface StartInferenceDialogProps {
    close: () => void;
    model: SchemaModel;
}

export const StartInferenceDialog = ({ close, model }: StartInferenceDialogProps) => {
    const [backend, setBackend] = useState<string>(getDefaultbackend(model));

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
            <Heading>Select your inference backend</Heading>
            <Divider />
            <Content>
                <Suspense fallback={<ProgressCircle aria-label='Loading backends' isIndeterminate size='S' />}>
                    <BackendSelection model={model} backend={backend} setBackend={setBackend} />
                </Suspense>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={onStart}>
                    Start
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
