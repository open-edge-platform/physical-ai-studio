import { Button, Content, ContextualHelp, Flex, Heading, Text } from '@geti-ui/ui';
import { createSearchParams, useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { SchemaModel, SchemaModelExportAction } from '../../../api/openapi-spec';
import { paths } from '../../../paths';
import { getDefaultInferenceDevice } from '../backend-selection/backend-selection';
import { ModelVariant } from './variant-selection';

export const RunVariant = ({
    model,
    backend,
    variant,
}: {
    model: SchemaModel;
    backend: SchemaModelExportAction['backend'];
    variant: ModelVariant;
}) => {
    const navigate = useNavigate();
    const runnable = backend === 'openvino' || backend === 'torch';
    const {
        data: devices = [],
        isPending,
        isError,
        refetch,
    } = $api.useQuery('get', '/api/system/devices/inference', {}, { enabled: runnable });
    const device = getDefaultInferenceDevice(devices, backend);
    const unavailable = !runnable
        ? 'This export cannot run in Studio.'
        : isError
          ? 'Could not load inference devices.'
          : !isPending && !device
            ? 'No compatible inference device is available.'
            : null;
    return (
        <Flex gap='size-100' alignItems='center'>
            <Button
                variant='accent'
                isDisabled={!runnable || isPending || isError || !device || !model.id}
                onPress={() => {
                    if (!device || !model.id || !runnable) return;
                    navigate({
                        pathname: paths.project.models.inference({
                            project_id: model.project_id,
                            model_id: model.id,
                            backend,
                        }),
                        search: createSearchParams({ device: device.device, variant }).toString(),
                    });
                }}
            >
                Run
            </Button>
            {unavailable && (
                <ContextualHelp aria-label='Why Run is unavailable' variant='info'>
                    <Heading>Studio inference</Heading>
                    <Content>
                        <Text>{unavailable}</Text>
                    </Content>
                </ContextualHelp>
            )}
            {runnable && isError && (
                <Button variant='secondary' onPress={() => void refetch()}>
                    Retry devices
                </Button>
            )}
        </Flex>
    );
};
