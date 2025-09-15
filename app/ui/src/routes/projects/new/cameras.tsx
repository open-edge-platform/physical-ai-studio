import { useState } from 'react';

import {
    ActionButton,
    Button,
    Checkbox,
    Flex,
    Form,
    Item,
    Key,
    NumberField,
    Picker,
    Section,
    TextField,
    Well,
} from '@geti/ui';
import { DeleteOutline } from '@geti/ui/icons';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { SchemaCamera, SchemaCameraConfig } from '../../../api/openapi-spec';
import { createEmptyCamera, useNewProject } from './new-project.provider';

interface CameraEditProps {
    availableCameras: SchemaCamera[];
    config: SchemaCameraConfig;
    updateConfig: (config: SchemaCameraConfig) => void;
    deleteCamera: () => void;
}

const CameraEdit = ({ config, availableCameras, updateConfig, deleteCamera }: CameraEditProps) => {
    const realSenseCameras = availableCameras.filter((c) => c.type == 'RealSense');
    const openCVCameras = availableCameras.filter((c) => c.type == 'OpenCV');

    const changeCamera = (id: Key | null) => {
        const selectedCamera = availableCameras.find((m) => m.id === id);
        if (selectedCamera) {
            updateConfig({
                ...config,
                id: selectedCamera.id,
                type: selectedCamera.type,
                use_depth: selectedCamera.type === 'RealSense',
                fps: selectedCamera.default_stream_profile.fps,
                width: selectedCamera.default_stream_profile.width,
                height: selectedCamera.default_stream_profile.height,
            });
        }
    };
    return (
        <Well position={'relative'}>
            <ActionButton
                aria-label='Icon only'
                onPress={deleteCamera}
                top={'size-100'}
                right={'size-200'}
                position={'absolute'}
            >
                <DeleteOutline />
            </ActionButton>
            <Form>
                <TextField label='name' value={config.name} onChange={(name) => updateConfig({ ...config, name })} />
                <Picker label='camera' selectedKey={config.id} onSelectionChange={changeCamera}>
                    <Section title='OpenCV'>
                        {openCVCameras.map((camera) => (
                            <Item key={camera.id}>{camera.name}</Item>
                        ))}
                    </Section>
                    <Section title='RealSense'>
                        {realSenseCameras.map((camera) => (
                            <Item key={camera.id}>{camera.name}</Item>
                        ))}
                    </Section>
                </Picker>
                <NumberField label='fps' value={config.fps} onChange={(fps) => updateConfig({ ...config, fps })} />
                <NumberField
                    label='width'
                    value={config.width}
                    onChange={(width) => updateConfig({ ...config, width })}
                />
                <NumberField
                    label='height'
                    value={config.height}
                    onChange={(height) => updateConfig({ ...config, height })}
                />
                <Checkbox
                    isSelected={config.use_depth}
                    isDisabled={config.type !== 'RealSense'}
                    onChange={(use_depth) => updateConfig({ ...config, use_depth })}
                >
                    Use depth
                </Checkbox>
            </Form>
        </Well>
    );
};

type CameraConfig = SchemaCameraConfig & { uuid: string };
export const CamerasView = () => {
    const { project, setProject } = useNewProject();
    const { data: availableCameras } = $api.useQuery('get', '/api/hardware/cameras');
    const [cameras, setCameras] = useState<CameraConfig[]>(
        project.cameras.map((camera) => {
            return {
                ...camera,
                uuid: uuidv4(),
            };
        })
    );

    const updateCamera = (uuid: string, camera: SchemaCameraConfig) => {
        const new_cameras = cameras.map((cam) => (cam.uuid == uuid ? { ...camera, uuid } : cam));
        setCameras(new_cameras);
        updateProjectFromCameras(new_cameras);
    };

    const updateProjectFromCameras = (new_cameras: CameraConfig[]) => {
        setProject({
            ...project,
            cameras: new_cameras.map((config) => {
                const { uuid, ...cameraConfig } = config;
                return cameraConfig;
            }),
        });
    };

    const addCamera = () => {
        const new_cameras = [...cameras, { ...createEmptyCamera({}), uuid: uuidv4() }];
        setCameras(new_cameras);
        updateProjectFromCameras(new_cameras);
    };

    const deleteCamera = (uuid: string) => {
        const new_cameras = cameras.filter((c) => c.uuid !== uuid);
        setCameras(new_cameras);
        updateProjectFromCameras(new_cameras);
    };

    return (
        <Flex direction={'column'} gap='size-100'>
            <Button alignSelf={'end'} onPress={addCamera}>
                Add Camera
            </Button>
            {(cameras ?? []).map((c) => (
                <CameraEdit
                    key={c.uuid}
                    config={c}
                    availableCameras={availableCameras ?? []}
                    updateConfig={(config) => updateCamera(c.uuid, config)}
                    deleteCamera={() => deleteCamera(c.uuid)}
                />
            ))}
        </Flex>
    );
};
