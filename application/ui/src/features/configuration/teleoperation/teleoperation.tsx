import { Suspense, useState } from 'react';

import {
    Button,
    ButtonGroup,
    ComboBox,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    IllustratedMessage,
    Item,
    Loading,
    Picker,
    Section,
    TextField,
    View,
} from '@geti/ui';

import { $api, fetchClient } from '../../../api/client';
import { SchemaDatasetOutput, SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { paths } from '../../../router';
import { useProjectId } from '../../projects/use-project';

interface TeleoperationSetupProps {
    onDone: (config: SchemaTeleoperationConfig | undefined) => void;
    dataset: SchemaDatasetOutput;
}

export const TeleoperationSetup = ({ dataset, onDone }: TeleoperationSetupProps) => {
    const { project_id } = useProjectId();
    const { data: projectTasks } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/tasks', {
        params: {
            path: { project_id },
        },
    });

    const initialTask = Object.values(projectTasks).flat()[0];

    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: {
            path: {
                project_id,
            },
        },
    });
    const [environmentId, setEnvironmentId] = useState<string | undefined>(environments[0]?.id);
    const [task, setTask] = useState<string>(initialTask);

    const isValid = () => {
        const taskValid = task !== '';
        return taskValid && environmentId !== undefined;
    };

    const onCancel = () => {
        onDone(undefined);
    };

    const onStart = async () => {
        const { data: environment } = await fetchClient.request(
            'get',
            '/api/projects/{project_id}/environments/{environment_id}',
            {
                params: {
                    path: {
                        environment_id: environmentId!,
                        project_id,
                    },
                },
            }
        );

        if (environment === undefined) {
            return;
        }

        onDone({
            dataset,
            task,
            environment,
        });
    };

    if (environmentId === undefined) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <Content>Currently there has not been a environment setup yet.</Content>
                    <Heading>No environment set up yet.</Heading>
                    <View margin={'size-100'}>
                        <Button variant='accent' href={paths.project.environments.new({ project_id })}>
                            Setup environment
                        </Button>
                    </View>
                </IllustratedMessage>
            </Flex>
        );
    }

    return (
        <View>
            <TextField
                validationState={dataset.name == '' ? 'invalid' : 'valid'}
                isRequired
                label='Dataset Name'
                width={'100%'}
                value={dataset.name}
                isDisabled={true}
            />
            <Picker
                items={environments}
                selectedKey={environmentId}
                label='Environment'
                onSelectionChange={(m) => setEnvironmentId(m === null ? undefined : m.toString())}
                flex={1}
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>
            <ComboBox
                validationState={task === '' ? 'invalid' : 'valid'}
                isRequired
                width={'100%'}
                label='Task'
                allowsCustomValue
                inputValue={task}
                onInputChange={setTask}
            >
                {Object.keys(projectTasks).map((datasetName) => (
                    <Section key={datasetName} title={datasetName}>
                        {projectTasks[datasetName].map((taskName) => (
                            <Item key={`${datasetName}-${taskName}`}>{taskName}</Item>
                        ))}
                    </Section>
                ))}
            </ComboBox>
            <Flex justifyContent={'end'}>
                <View paddingTop={'size-300'}>
                    <ButtonGroup>
                        <Button onPress={onCancel} variant='secondary'>
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

export const TeleoperationSetupModal = (
    close: (config: SchemaTeleoperationConfig | undefined) => void,
    dataset: SchemaDatasetOutput
) => {
    return (
        <Dialog>
            <Heading>Teleoperate Setup</Heading>
            <Divider />
            <Content>
                <Suspense fallback={<Loading mode='overlay' />}>
                    <TeleoperationSetup dataset={dataset} onDone={close} />
                </Suspense>
            </Content>
        </Dialog>
    );
};
