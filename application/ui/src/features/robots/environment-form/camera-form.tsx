import { useState } from 'react';

import { ActionButton, Button, Flex, Heading, Icon, Item, Picker, Text, TextField, View, Well } from '@geti-ui/ui';
import { Add, Close } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { SchemaProjectCamera } from '../../../api/types';
import { useProjectId } from '../../../features/projects/use-project';
import { CameraConfiguration, useEnvironmentForm, useSetEnvironmentForm } from './provider';

import classes from './form.module.scss';

export const CameraListItem = ({ camera, onRemove }: { camera: CameraConfiguration; onRemove: () => void }) => {
    const { project_id } = useProjectId();
    const camerasQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    const projectCamera = camerasQuery.data.find(({ id }) => id === camera.camera_id);

    if (projectCamera === undefined) {
        return <li>{camera.camera_id} - unknown</li>;
    }

    return (
        <li>
            <View backgroundColor={'gray-50'} padding='size-200' borderColor='gray-200' borderWidth='thick'>
                <Flex justifyContent='space-between' alignItems={'center'}>
                    <Flex direction='column'>
                        <span>{camera.name}</span>
                        <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>
                            {projectCamera.name}
                        </Text>
                    </Flex>

                    <ActionButton onPress={onRemove} UNSAFE_className={classes.actionButton}>
                        <Icon>
                            <Close />
                        </Icon>
                    </ActionButton>
                </Flex>
            </View>
        </li>
    );
};

const getAvailableCameras = (environmentCameras: Array<CameraConfiguration>, cameras: Array<SchemaProjectCamera>) => {
    const environmentCameraIds = environmentCameras.map(({ camera_id }) => camera_id);

    return cameras.filter((camera) => {
        // Don't allow adding the same camera twice
        return environmentCameraIds.includes(camera.id!) === false;
    });
};

export const AddCameraForm = ({
    onAddCamera,
    onCancel,
}: {
    onAddCamera: (camera: CameraConfiguration) => void;
    onCancel?: () => void;
}) => {
    const { project_id } = useProjectId();
    const camerasQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });
    const environment = useEnvironmentForm();

    const availableCameras = getAvailableCameras(environment.cameras, camerasQuery.data);

    const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
    const [name, setName] = useState('');

    if (availableCameras.length === 0) {
        return <span>No available cameras</span>;
    }

    return (
        <Flex direction='column' gap='size-100'>
            <Heading level={4}>Add camera</Heading>

            <Picker
                label='Camera'
                width='100%'
                selectedKey={selectedCameraId}
                onSelectionChange={(key) => {
                    if (key !== null && typeof key === 'string') {
                        setSelectedCameraId(key);
                        // Default the per-environment name to the camera's own name.
                        setName(availableCameras.find((camera) => camera.id === key)?.name ?? '');
                    }
                }}
            >
                {availableCameras.map((camera) => {
                    return (
                        <Item textValue={camera.name} key={camera.id}>
                            <Text>{camera.name}</Text>
                        </Item>
                    );
                })}
            </Picker>

            <TextField label='Name' width='100%' value={name} onChange={setName} />

            <Flex gap='size-100'>
                <Button
                    variant='secondary'
                    isDisabled={!selectedCameraId || name.length === 0}
                    onPress={() => {
                        if (selectedCameraId && name.length > 0) {
                            onAddCamera({ camera_id: selectedCameraId, name });
                        }
                    }}
                >
                    Add
                </Button>
                {onCancel && (
                    <Button variant='secondary' onPress={onCancel}>
                        Cancel
                    </Button>
                )}
            </Flex>
        </Flex>
    );
};

export const CameraForm = () => {
    const environmentForm = useEnvironmentForm();
    const setEnvironmentForm = useSetEnvironmentForm();

    const hasNoCameras = environmentForm.cameras.length === 0;
    const [isAdding, setIsAdding] = useState(hasNoCameras);

    return (
        <>
            {environmentForm.cameras.length > 0 && (
                <ul style={{ width: '100%' }}>
                    <Flex direction='column' gap='size-100' width='100%'>
                        {environmentForm.cameras.map((camera) => (
                            <CameraListItem
                                key={camera.camera_id}
                                camera={camera}
                                onRemove={() => {
                                    setEnvironmentForm((oldForm) => {
                                        return {
                                            ...oldForm,
                                            cameras: oldForm.cameras.filter(
                                                ({ camera_id }) => camera_id !== camera.camera_id
                                            ),
                                        };
                                    });
                                }}
                            />
                        ))}
                    </Flex>
                </ul>
            )}

            {isAdding ? (
                <Well
                    width='100%'
                    UNSAFE_style={{
                        backgroundColor: 'var(--spectrum-global-color-gray-200)',
                    }}
                >
                    <AddCameraForm
                        onAddCamera={(camera) => {
                            setEnvironmentForm((oldForm) => {
                                return { ...oldForm, cameras: [...oldForm.cameras, camera] };
                            });
                            setIsAdding(false);
                        }}
                        onCancel={
                            hasNoCameras
                                ? undefined
                                : () => {
                                      setIsAdding(false);
                                  }
                        }
                    />
                </Well>
            ) : (
                <Button
                    variant='secondary'
                    UNSAFE_className={classes.addNewButton}
                    width='100%'
                    onPress={() => {
                        setIsAdding(true);
                    }}
                >
                    <Icon marginEnd='size-50'>
                        <Add />
                    </Icon>
                    Camera
                </Button>
            )}
        </>
    );
};
