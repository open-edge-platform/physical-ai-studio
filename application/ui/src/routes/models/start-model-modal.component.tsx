import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Heading } from '@geti-ui/ui';
import { useNavigate } from 'react-router';

import { BackendSelection, defaultBackend } from '../../features/configuration/shared/backend-selection';
import { paths } from '../../router';

interface StartInferenceDialogProps {
    close: () => void;
    project_id: string;
    model_id: string;
}
export const StartInferenceDialog = ({ close, project_id, model_id }: StartInferenceDialogProps) => {
    const [backend, setBackend] = useState<string>(defaultBackend);

    const navigate = useNavigate();
    const onStart = () => {
        close();
        navigate(
            paths.project.models.inference({
                project_id,
                model_id,
                backend,
            })
        );
    };

    return (
        <Dialog>
            <Heading>Run model</Heading>
            <Divider />
            <Content>
                <BackendSelection backend={backend} setBackend={setBackend} />
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
