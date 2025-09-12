import { v4 as uuidv4 } from "uuid"
import { Well, Flex, TabList, Item, TabPanels, View, Form, TextField, NumberField, Picker, Section, Checkbox, Divider, Key, Button, ActionButton } from '@geti/ui';
import { $api } from "../../../api/client";
import { createEmptyCamera, useProjectDataContext } from "./project-config.provider";
import { SchemaCamera, SchemaCameraConfig } from '../../../api/openapi-spec';
import { useState } from 'react';
import { DeleteOutline } from "@geti/ui/icons";

interface CameraEditProps {
    availableCameras: SchemaCamera[];
    config: SchemaCameraConfig;
    updateConfig: (config: SchemaCameraConfig) => void;
    deleteCamera: () => void;
}

const CameraEdit = ({ config, availableCameras, updateConfig, deleteCamera }: CameraEditProps) => {
    const realSenseCameras = availableCameras.filter((c) => c.type == "RealSense")
    const openCVCameras = availableCameras.filter((c) => c.type == "OpenCV")

    const changeCamera = (id: Key | null) => {
        if (id) {
            const use_depth = !!realSenseCameras.find((m) => m.id === id)
            updateConfig({ ...config, id: id as string, use_depth })
        }
    }
    return (
        <Well position={'relative'}>
            <ActionButton aria-label="Icon only" onPress={deleteCamera} top={'size-100'} right={'size-200'} position={'absolute'}>
                <DeleteOutline />
            </ActionButton>
            <Form>
                <TextField label="name" value={config.name} onChange={(name) => updateConfig({ ...config, name })} />
                <Picker label="camera" selectedKey={config.id} onSelectionChange={changeCamera}>
                    <Section title="OpenCV">
                        {openCVCameras.map((camera) => (
                            <Item key={camera.id}>{camera.name}</Item>
                        ))}

                    </Section>
                    <Section title="RealSense">
                        {realSenseCameras.map((camera) => (
                            <Item key={camera.id}>{camera.name}</Item>
                        ))}
                    </Section>
                </Picker>
                <NumberField label="fps" value={config.fps} onChange={(fps) => updateConfig({ ...config, fps })} />
                <NumberField label="width" value={config.width} onChange={(width) => updateConfig({ ...config, width })} />
                <NumberField label="height" value={config.height} onChange={(height) => updateConfig({ ...config, height })} />
                <Checkbox isSelected={config.use_depth} onChange={(use_depth) => updateConfig({ ...config, use_depth })}>Use depth</Checkbox>
            </Form>
        </Well>
    )
}

type CameraConfig = SchemaCameraConfig & { uuid: string }
export const CamerasView = () => {
    const { project, setProject } = useProjectDataContext();
    const { data: availableCameras } = $api.useQuery('get', '/api/hardware/cameras');
    const [cameras, setCameras] = useState<CameraConfig[]>(project.cameras.map((camera) => {
        return {
            ...camera,
            uuid: uuidv4(),
        }
    }));

    const updateCamera = (uuid: string, camera: SchemaCameraConfig) => {
        const new_cameras = cameras.map((cam) => (cam.uuid == uuid ? { ...camera, uuid } : cam));
        setCameras(new_cameras)
        updateProjectFromCameras(new_cameras)
    }

    const updateProjectFromCameras = (cameras: CameraConfig[]) => {
        setProject({
            ...project,
            cameras: cameras.map((config) => {
                const { uuid, ...cameraConfig } = config;
                return cameraConfig;
            })
        });
    }

    const addCamera = () => {
        const new_cameras = [...cameras, { ...createEmptyCamera({}), uuid: uuidv4() }];
        setCameras(new_cameras);
        updateProjectFromCameras(new_cameras);
    }

    const deleteCamera = (uuid: string) => {
        const new_cameras = cameras.filter((c) => c.uuid !== uuid)
        setCameras(new_cameras);
        updateProjectFromCameras(new_cameras)
    }

    return (
        <>
            <Button onPress={addCamera}>Add Camera</Button>
            {(cameras ?? []).map((c) => (
                <>
                    <CameraEdit
                        key={c.uuid}
                        config={c}
                        availableCameras={availableCameras ?? []}
                        updateConfig={(config) => updateCamera(c.uuid, config)}
                        deleteCamera={() => deleteCamera(c.uuid)}
                    />
                    <Divider />
                </>
            ))}
        </>
    );
}