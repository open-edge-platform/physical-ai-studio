import { useState } from 'react';

import { ActionButton, Button, Flex, Heading, Icon, Item, Picker, Text, View, Well } from '@geti/ui';
import { Add, Close } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { SchemaProjectCamera } from '../../../api/types';
import { useProjectId } from '../../../features/projects/use-project';
import { useEnvironmentForm, useSetEnvironmentForm } from './provider';

import classes from './form.module.scss';

export const CameraListItem = ({ cameraId, onRemove }: { cameraId: string; onRemove: () => void }) => {
    const { project_id } = useProjectId();
    const camerasQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });

    const camera = camerasQuery.data.find(({ id }) => id === cameraId);

    if (camera === undefined) {
        return <li>{cameraId} - unknown</li>;
    }

    return (
        <li>
            <View backgroundColor={'gray-50'} padding='size-200' borderColor='gray-200' borderWidth='thick'>
                <Flex justifyContent='space-between' alignItems={'center'}>
                    {camera.name}

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

const getAvailableCameras = (environmentCameraIds: Array<string>, cameras: Array<SchemaProjectCamera>) => {
    const environmentCameras = environmentCameraIds.map((id) => {
        return cameras.find((camera) => camera.id === id);
    });

    return cameras.filter((camera) => {
        // Don't allow adding the same camera twice
        if (environmentCameraIds.includes(camera.id!)) {
            return false;
        }

        // Don't allow adding duplicated camera names
        return environmentCameras.some((environmentCamera) => environmentCamera?.name === camera.name) === false;
    });
};

export const AddCameraForm = ({
    onAddCamera,
    onCancel,
}: {
    onAddCamera: (cameraId: string) => void;
    onCancel?: () => void;
}) => {
    const { project_id } = useProjectId();
    const camerasQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/cameras', {
        params: { path: { project_id } },
    });
    const environment = useEnvironmentForm();

    const availableCameras = getAvailableCameras(environment.camera_ids, camerasQuery.data);

    const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);

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

            <Flex gap='size-100'>
                <Button
                    variant='secondary'
                    onPress={() => {
                        if (selectedCameraId) {
                            onAddCamera(selectedCameraId);
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

    const hasNoCameras = environmentForm.camera_ids.length === 0;
    const [isAdding, setIsAdding] = useState(hasNoCameras);

    return (
        <>
            <ul style={{ width: '100%' }}>
                <Flex direction='column' gap='size-100' width='100%'>
                    {environmentForm.camera_ids.map((id) => (
                        <CameraListItem
                            key={id}
                            cameraId={id}
                            onRemove={() => {
                                setEnvironmentForm((oldForm) => {
                                    return {
                                        ...oldForm,
                                        camera_ids: oldForm.camera_ids.filter((cameraId) => cameraId !== id),
                                    };
                                });
                            }}
                        />
                    ))}
                </Flex>
            </ul>

            {isAdding ? (
                <Well
                    width='100%'
                    UNSAFE_style={{
                        backgroundColor: 'var(--spectrum-global-color-gray-200)',
                    }}
                >
                    <AddCameraForm
                        onAddCamera={(cameraId) => {
                            setEnvironmentForm((oldForm) => {
                                return { ...oldForm, camera_ids: [...oldForm.camera_ids, cameraId] };
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
