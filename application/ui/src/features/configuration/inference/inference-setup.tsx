import { Suspense, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    Item,
    Key,
    Loading,
    Picker,
    TabList,
    TabPanels,
    Tabs,
    View,
} from '@geti/ui';

import { $api, fetchClient } from '../../../api/client';
import { SchemaInferenceConfig } from '../../../api/openapi-spec';
import { useProjectId } from '../../projects/use-project';
import { availableBackends, BackendSelection } from '../shared/backend-selection';

interface InferenceSetupProps {
    onDone: (config: SchemaInferenceConfig | undefined) => void;
    model_id: string;
}

export const InferenceSetup = ({ model_id, onDone }: InferenceSetupProps) => {
    const { project_id } = useProjectId();
    const { data: model } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { query: { uuid: model_id } },
    });
    const {data: environments} = $api.useSuspenseQuery('get','/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    })

    const [environmentId, setEnvironmentId] = useState<string | undefined>(environments[0]?.id);
    const [backend, setBackend] = useState<string>(availableBackends[0].id);
    const isValid = () => {
        return environmentId !== undefined;
    };

    const onBack = () => {
        onDone(undefined);
    };

    const onStart = async () => {
        if (environmentId === undefined) {
            return
        }
        const { data: environment } = await fetchClient.request("get", "/api/projects/{project_id}/environments/{environment_id}",
            {
                params: {
                    path: {
                        environment_id: environmentId,
                        project_id
                    }
                }
            }
        )

        if (environment === undefined) {
            return
        }

        onDone({
            backend,
            model,
            task_index: 0,
            environment: environment,
        })
    }

    return (
        <View>
            <View height={'330px'}>
                <Picker
                    items={environments}
                    selectedKey={environmentId}
                    label='Environment'
                    onSelectionChange={(m) => setEnvironmentId(m === null ? undefined : m.toString())}
                    flex={1}
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>
            </View>
            <Flex justifyContent={'space-between'}>
                <View>
                    <BackendSelection
                        backend={backend}
                        setBackend={setBackend}
                    />
                </View>
                <View paddingTop={'size-300'}>
                    <ButtonGroup>
                        <Button onPress={onBack} variant='secondary'>
                            Cancel
                        </Button>
                        <Button onPress={onStart} isDisabled={!isValid()}>
                            Start
                        </Button>
                    </ButtonGroup>
                </View>
            </Flex>
        </View>
    );
};

export const InferenceSetupModal = (close: (config: SchemaInferenceConfig | undefined) => void, model_id: string) => {
    return (
        <Dialog>
            <Heading>Inference Setup</Heading>
            <Divider />
            <Content>
                <Suspense fallback={<Loading mode='overlay' />}>
                    <InferenceSetup model_id={model_id} onDone={close} />
                </Suspense>
            </Content>
        </Dialog>
    );
};
