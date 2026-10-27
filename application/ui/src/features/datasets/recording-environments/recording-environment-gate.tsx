import { ReactNode, useEffect, useRef, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Heading,
    Item,
    Loading,
    Picker,
    Text,
} from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { SchemaRecordingEnvironment } from '../../../api/openapi-spec';
import { paths } from '../../../paths';

interface RecordingEnvironmentGateProps {
    project_id: string;
    dataset_id: string;
    onCancel: () => void;
    children: (environment: SchemaRecordingEnvironment) => ReactNode;
}

const useAssociateRecordingEnvironment = (dataset_id: string) => {
    const queryClient = useQueryClient();
    const queryKey = $api.queryOptions('get', '/api/dataset/{dataset_id}/recording-environments', {
        params: { path: { dataset_id } },
    }).queryKey;

    return $api.useMutation('post', '/api/dataset/{dataset_id}/recording-environments', {
        onSuccess: (environment) => {
            // Make the association available before navigating or mounting the runtime session.
            queryClient.setQueryData<SchemaRecordingEnvironment[]>(queryKey, (previous = []) => [
                ...previous.filter(({ id }) => id !== environment.id),
                environment,
            ]);
        },
    });
};

interface AssociateEnvironmentProps {
    dataset_id: string;
    environment_id: string;
    onCancel: () => void;
}

const AssociateSingleEnvironment = ({ dataset_id, environment_id, onCancel }: AssociateEnvironmentProps) => {
    const { mutate, isError } = useAssociateRecordingEnvironment(dataset_id);
    const started = useRef(false);
    const associate = () => mutate({ params: { path: { dataset_id } }, body: { environment_id, camera_mappings: [] } });

    useEffect(() => {
        // Effects can run twice in StrictMode; associate only once per mounted guard.
        if (started.current) return;
        started.current = true;
        mutate({ params: { path: { dataset_id } }, body: { environment_id, camera_mappings: [] } });
    }, [dataset_id, environment_id, mutate]);

    if (!isError) return <Loading mode='overlay' />;

    return (
        <DialogContainer onDismiss={onCancel}>
            <Dialog size='S'>
                <Heading>Could not use recording environment</Heading>
                <Content>Please try again.</Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={onCancel}>
                        Cancel
                    </Button>
                    <Button variant='accent' onPress={associate}>
                        Try again
                    </Button>
                </ButtonGroup>
            </Dialog>
        </DialogContainer>
    );
};

interface SelectRecordingEnvironmentProps {
    project_id: string;
    dataset_id: string;
    onCancel: () => void;
    environments: { id: string; name: string }[];
}

export const SelectRecordingEnvironment = ({
    project_id,
    dataset_id,
    environments,
    onCancel,
}: SelectRecordingEnvironmentProps) => {
    const [environmentId, setEnvironmentId] = useState<string>();
    const association = useAssociateRecordingEnvironment(dataset_id);

    return (
        <DialogContainer onDismiss={onCancel}>
            <Dialog size='S'>
                <Heading>
                    {environments.length === 0
                        ? 'No recording environment available'
                        : 'Select a recording environment'}
                </Heading>
                <Content>
                    {environments.length === 0 ? (
                        <Text>Create an environment before recording episodes for this dataset.</Text>
                    ) : (
                        <Picker
                            label='Environment'
                            items={environments}
                            selectedKey={environmentId ?? null}
                            onSelectionChange={(id) => setEnvironmentId(id?.toString())}
                            isDisabled={association.isPending}
                        >
                            {(environment) => <Item key={environment.id}>{environment.name}</Item>}
                        </Picker>
                    )}
                    {association.isError && <Text>Could not associate this environment. Please try again.</Text>}
                </Content>
                <ButtonGroup>
                    <Button variant='secondary' onPress={onCancel}>
                        Cancel
                    </Button>
                    {environments.length === 0 ? (
                        <Button variant='accent' href={paths.project.environments.new({ project_id })}>
                            Create environment
                        </Button>
                    ) : (
                        <Button
                            variant='accent'
                            isDisabled={environmentId === undefined}
                            isPending={association.isPending}
                            onPress={() => {
                                if (environmentId === undefined) return;
                                association.mutate({
                                    params: { path: { dataset_id } },
                                    body: { environment_id: environmentId, camera_mappings: [] },
                                });
                            }}
                        >
                            Use environment
                        </Button>
                    )}
                </ButtonGroup>
            </Dialog>
        </DialogContainer>
    );
};

export const RecordingEnvironmentGate = ({
    project_id,
    dataset_id,
    onCancel,
    children,
}: RecordingEnvironmentGateProps) => {
    const { data: recordingEnvironments } = $api.useSuspenseQuery(
        'get',
        '/api/dataset/{dataset_id}/recording-environments',
        { params: { path: { dataset_id } } }
    );
    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });
    const recordingEnvironment = recordingEnvironments.find(({ environment_id }) =>
        environments.some(({ id }) => id === environment_id)
    );

    if (recordingEnvironment !== undefined) return children(recordingEnvironment);

    if (environments.length === 1) {
        return (
            <AssociateSingleEnvironment
                key={`${dataset_id}:${environments[0].id}`}
                dataset_id={dataset_id}
                environment_id={environments[0].id}
                onCancel={onCancel}
            />
        );
    }

    return (
        <SelectRecordingEnvironment
            project_id={project_id}
            dataset_id={dataset_id}
            environments={environments}
            onCancel={onCancel}
        />
    );
};
