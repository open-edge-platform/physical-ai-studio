import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaModel, SchemaModelExportAction } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS } from '../inference-backends';

export const RemoveBackendDialog = ({
    model,
    backend,
    close,
}: {
    model: SchemaModel;
    backend: SchemaModelExportAction['backend'];
    close: () => void;
}) => {
    const remove = $api.useMutation('delete', '/api/models/{model_id}/exports/{backend}', {
        meta: {
            invalidates: [
                ['get', '/api/models/{model_id}', { params: { path: { model_id: model.id! } } }],
                ['get', '/api/projects/{project_id}/models', { params: { path: { project_id: model.project_id } } }],
            ],
        },
        onSuccess: close,
    });
    return (
        <Dialog>
            <Heading>Remove model</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>Permanently remove the {INFERENCE_BACKENDS[backend].label} export?</Text>
                    <Text>The training checkpoint, training history, and other backends are kept.</Text>
                    <Text>Unload this model from inference and wait for export jobs to finish before removing it.</Text>
                    {remove.error && (
                        <div role='alert'>{getApiErrorMessage(remove.error) ?? 'Could not remove the model.'}</div>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' isDisabled={remove.isPending} onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='negative'
                    isPending={remove.isPending}
                    onPress={() => remove.mutate({ params: { path: { model_id: model.id!, backend } } })}
                >
                    Remove model
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
