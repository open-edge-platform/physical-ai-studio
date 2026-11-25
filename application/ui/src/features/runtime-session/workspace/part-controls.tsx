import { createContext, ReactNode, Suspense, useContext, useState } from 'react';

import {
    AlertDialog,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Heading,
    Item,
    Picker,
    ProgressCircle,
    Text,
    View,
} from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaEnvironmentDraft, SchemaModel } from '../../../api/openapi-spec';
import {
    BackendSelection,
    defaultBackend,
    getDefaultInferenceDevice,
    getSupportedInferenceDevices,
    InferenceDevice,
} from '../../models/backend-selection/backend-selection';
import { useProject, useProjectId } from '../../projects/use-project';
import { SessionPartName, useRuntimeSession } from '../runtime-session-provider';
import { PendingConfirmation, SessionCommand, usePartCommand } from './use-part-command';

export type PartAction = 'reload' | 'swap' | 'unload';

interface Choice {
    id: string;
    name: string;
}

/** Pick an environment, dataset or model to load in place of the current one. */
const SwapDialog = ({
    title,
    choices,
    current,
    onPick,
    close,
}: {
    title: string;
    choices: Choice[];
    current: string | null | undefined;
    onPick: (id: string) => void;
    close: () => void;
}) => {
    const [selected, setSelected] = useState<string | null>(current ?? null);

    return (
        <Dialog>
            <Heading>{title}</Heading>
            <Divider />
            <Content>
                <Picker
                    label={title}
                    width='100%'
                    selectedKey={selected}
                    onSelectionChange={(key) => setSelected(key === null ? null : String(key))}
                    items={choices}
                >
                    {(choice) => <Item key={choice.id}>{choice.name}</Item>}
                </Picker>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    isDisabled={selected === null}
                    onPress={() => {
                        close();
                        if (selected !== null) {
                            onPick(selected);
                        }
                    }}
                >
                    Load
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

/**
 * Pick a model (newest first, with its policy, dataset and when it was trained), then the
 * backend and device to run it on; these start at the loaded model's, else the defaults.
 */
const LoadModelDialog = ({
    models,
    datasetNames,
    current,
    onPick,
    close,
}: {
    models: SchemaModel[];
    datasetNames: Map<string, string>;
    current: { model_id: string; backend: string; device: string } | null | undefined;
    onPick: (model_id: string, backend: string, device: string) => void;
    close: () => void;
}) => {
    const [selected, setSelected] = useState<string | null>(current?.model_id ?? null);
    const [inferenceDevice, setInferenceDevice] = useState<InferenceDevice | undefined>(
        current ? { backend: current.backend as InferenceDevice['backend'], device: current.device } : undefined
    );
    const { data: inferenceDevices = [], isLoading } = $api.useQuery('get', '/api/system/devices/inference');

    // A model without an export has nothing to run.
    const choices = models
        .filter((model) => model.id && model.available_backends.length > 0)
        .toSorted((a, b) => new Date(b.created_at ?? 0).getTime() - new Date(a.created_at ?? 0).getTime());
    const picked = choices.find((model) => model.id === selected);
    const backend =
        inferenceDevice?.backend ??
        (picked?.available_backends.includes(defaultBackend) ? defaultBackend : picked?.available_backends.at(0)) ??
        defaultBackend;
    const device =
        getSupportedInferenceDevices(inferenceDevices, backend).find(
            (candidate) => candidate.device === inferenceDevice?.device
        ) ?? getDefaultInferenceDevice(inferenceDevices, backend);

    return (
        <Dialog>
            <Heading>Model</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Picker
                        label='Model'
                        width='100%'
                        selectedKey={selected}
                        onSelectionChange={(key) => {
                            setSelected(key === null ? null : String(key));
                            setInferenceDevice(undefined);
                        }}
                        items={choices}
                    >
                        {(model) => (
                            <Item key={model.id} textValue={model.name}>
                                <Text>{model.name}</Text>
                                <Text slot='description'>
                                    {[
                                        model.policy.toUpperCase(),
                                        (model.dataset_id && datasetNames.get(model.dataset_id)) ?? 'No dataset',
                                        model.created_at && new Date(model.created_at).toLocaleString(),
                                    ]
                                        .filter(Boolean)
                                        .join(' · ')}
                                </Text>
                            </Item>
                        )}
                    </Picker>
                    {picked !== undefined && (
                        <Suspense fallback={<ProgressCircle aria-label='Loading backends' isIndeterminate size='S' />}>
                            <BackendSelection
                                model={picked}
                                inferenceDevice={device}
                                inferenceDevices={inferenceDevices}
                                setInferenceDevice={setInferenceDevice}
                            />
                        </Suspense>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    isDisabled={picked?.id === undefined || device === undefined || isLoading}
                    onPress={() => {
                        close();
                        if (picked?.id !== undefined && device !== undefined) {
                            onPick(picked.id, backend, device.device);
                        }
                    }}
                >
                    Load
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

/** Confirm a load that unloads parts which no longer fit, listing why. */
const ConfirmUnload = ({ confirmation }: { confirmation: PendingConfirmation }) => (
    <AlertDialog
        title='Unload parts that no longer fit?'
        variant='warning'
        primaryActionLabel='Unload and continue'
        secondaryActionLabel='Cancel'
        onPrimaryAction={confirmation.confirm}
        onSecondaryAction={confirmation.cancel}
    >
        <Flex direction='column' gap='size-100'>
            {confirmation.parts.map((part) => (
                <View key={`${part.part}-${part.name}`}>
                    <Text>
                        The {part.part} <strong>{part.name}</strong> will be unloaded:
                    </Text>
                    <ul>
                        {part.reasons.map((reason) => (
                            <li key={reason}>{reason}</li>
                        ))}
                    </ul>
                </View>
            ))}
        </Flex>
    </AlertDialog>
);

interface PartControlsValue {
    /** The part a command or the page's own load is busy with. */
    pendingPart: SessionPartName | null;
    act: (part: SessionPartName, action: PartAction) => void;
}

const PartControlsContext = createContext<PartControlsValue | null>(null);

/**
 * Reload, swap (a picker of the project's environments, datasets or models) and unload, for
 * anything in the workspace that offers them: the parts panel, and a mode saying what it misses.
 *
 * Swapping the environment can change the feature spec; the session then refuses, and the
 * confirmation lists what would be unloaded before the load is sent again.
 */
export const PartControlsProvider = ({ children }: { children: ReactNode }) => {
    const { project_id } = useProjectId();
    const project = useProject();
    const { state, loading } = useRuntimeSession();
    const { run, confirmation } = usePartCommand();
    const [commandPart, setCommandPart] = useState<SessionPartName | null>(null);
    const [swapping, setSwapping] = useState<SessionPartName | null>(null);

    const { data: environments = [] } = $api.useQuery('get', '/api/projects/{project_id}/environments', {
        params: { path: { project_id } },
    });
    const { data: models = [] } = $api.useQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id } },
    });
    const datasets = project.datasets ?? [];
    const dataset = state?.dataset;
    const model = state?.model;

    const runFor = async (part: SessionPartName, command: SessionCommand) => {
        setCommandPart(part);
        try {
            await run(command);
        } finally {
            setCommandPart(null);
        }
    };

    const loadEnvironment = (environment_id: string) => {
        const saved = environments.find((candidate) => candidate.id === environment_id);
        if (saved !== undefined) {
            const draft: SchemaEnvironmentDraft = {
                id: saved.id,
                name: saved.name,
                robots: saved.robots ?? [],
                cameras: saved.cameras ?? [],
            };
            void runFor('environment', { command: 'load_environment', environment: draft, unload_incompatible: false });
        }
    };

    const reload = (part: SessionPartName) => {
        if (part === 'environment') {
            // The environment connects its devices again in place; it stays loaded if that fails.
            void runFor('environment', { command: 'reload_environment' });
        } else if (part === 'dataset' && dataset) {
            // Loading the same dataset again finishes the recording (copies it back) and reopens it.
            void runFor('dataset', { command: 'load_dataset', dataset_id: dataset.dataset_id });
        } else if (part === 'model' && model) {
            // Loading it again starts its process again.
            void runFor('model', {
                command: 'load_model',
                model_id: model.model_id,
                backend: model.backend,
                device: model.device,
            });
        }
    };

    const act = (part: SessionPartName, action: PartAction) => {
        if (action === 'swap') {
            setSwapping(part);
        } else if (action === 'unload') {
            const command = {
                environment: 'unload_environment',
                dataset: 'unload_dataset',
                model: 'unload_model',
            } as const;
            void runFor(part, { command: command[part] });
        } else {
            reload(part);
        }
    };

    const close = () => setSwapping(null);

    return (
        <PartControlsContext.Provider value={{ pendingPart: commandPart ?? loading, act }}>
            {children}
            <DialogContainer onDismiss={close}>
                {swapping === 'environment' && (
                    <SwapDialog
                        title='Environment'
                        choices={environments.flatMap((choice) =>
                            choice.id ? [{ id: choice.id, name: choice.name }] : []
                        )}
                        current={state?.environment?.environment_id}
                        onPick={loadEnvironment}
                        close={close}
                    />
                )}
                {swapping === 'dataset' && (
                    <SwapDialog
                        title='Dataset'
                        choices={datasets.flatMap((choice) =>
                            choice.id ? [{ id: choice.id, name: choice.name }] : []
                        )}
                        current={dataset?.dataset_id}
                        onPick={(dataset_id) => void runFor('dataset', { command: 'load_dataset', dataset_id })}
                        close={close}
                    />
                )}
                {swapping === 'model' && (
                    <LoadModelDialog
                        models={models}
                        datasetNames={
                            new Map(datasets.flatMap((choice) => (choice.id ? [[choice.id, choice.name]] : [])))
                        }
                        current={model}
                        onPick={(model_id, backend, device) =>
                            void runFor('model', { command: 'load_model', model_id, backend, device })
                        }
                        close={close}
                    />
                )}
            </DialogContainer>
            <DialogContainer onDismiss={() => confirmation?.cancel()}>
                {confirmation !== null && <ConfirmUnload confirmation={confirmation} />}
            </DialogContainer>
        </PartControlsContext.Provider>
    );
};

export const usePartControls = () => {
    const context = useContext(PartControlsContext);
    if (context === null) {
        throw new Error('usePartControls must be used within a PartControlsProvider');
    }
    return context;
};
