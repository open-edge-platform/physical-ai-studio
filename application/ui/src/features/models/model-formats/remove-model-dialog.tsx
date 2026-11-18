import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Heading, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaExportVariantDetail, SchemaModel, SchemaModelExportAction } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS } from '../inference-backends';

export const RemoveModelDialog = ({
    model,
    backend,
    variant,
    close,
}: {
    model: SchemaModel;
    backend: SchemaModelExportAction['backend'];
    variant: SchemaExportVariantDetail;
    close: () => void;
}) => {
    const remove = $api.useMutation('delete', '/api/models/{model_id}/exports/{backend}/variants/{variant}', {
        meta: {
            invalidates: [
                ['get', '/api/models/{model_id}', { params: { path: { model_id: model.id! } } }],
                ['get', '/api/projects/{project_id}/models', { params: { path: { project_id: model.project_id } } }],
            ],
        },
        onSuccess: close,
    });
    const megabytes = variant.size_bytes / 1024 ** 2;
    const size = megabytes >= 1024 ? `${(megabytes / 1024).toFixed(1)} GB` : `${megabytes.toFixed(1)} MB`;
    const label = INFERENCE_BACKENDS[backend].label;
    return (
        <Dialog>
            <Heading>Remove model</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        Permanently remove {label} · {variant.id === 'int8' ? 'INT8' : 'Original'}? Other models stay
                        available.
                    </Text>
                    <Text>
                        Approximately {size} will be freed. The training checkpoint, training history, and other
                        backends are kept.
                    </Text>
                    <Text>Unload this model from inference and wait for export jobs to finish before removing it.</Text>
                    {remove.error && (
                        <div role='alert'>
                            {getApiErrorMessage(remove.error) ?? 'Could not remove the model. Please try again.'}
                        </div>
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
                    onPress={() => {
                        remove.mutate({ params: { path: { model_id: model.id!, backend, variant: variant.id } } });
                    }}
                >
                    Remove model
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
