import { View, Form, Heading, TextField, Picker, Section, ActionButton, Text, Flex, Key, Well, Tabs, TabList, TabPanels, Item } from '@geti/ui'
import { useProject } from '../../projects/project.provider';
import { useLocation, useParams } from 'react-router';
import { useSearchParams } from 'react-router-dom';
import { useState } from 'react';
import { SchemaCameraConfig, SchemaCamera, SchemaProjectConfig, SchemaRobotConfig, SchemaRobotPortInfo } from '../../../api/openapi-spec';
import { $api } from '../../../api/client';


interface CameraSetupProps {
    camera: SchemaCameraConfig;
    availableCameras: SchemaCamera[];
    updateCamera: (name: string, id: string, oldId: string) => void;
}
const CameraSetup = ({ camera, availableCameras, updateCamera }: CameraSetupProps) => {
    const camerasConnectedOfType = availableCameras.filter((m) => m.type === camera.type);

    const onSelection = (key: Key | null) => {
        if (key) {
            updateCamera(camera.name, String(key), camera.id);
        }
    }

    return (
        <>
            <Text>Camera: {camera.name}</Text>
            <>Some preview window here...</>
            <Picker label='camera' selectedKey={camera.id} onSelectionChange={onSelection}>
                {camerasConnectedOfType.map((camera) => (
                    <Item key={camera.id}>{camera.name}</Item>
                ))}
            </Picker>
        </>
    );
}

interface RobotSetupProps {
    config: SchemaRobotConfig,
    portInfos: SchemaRobotPortInfo[]
}
const RobotSetup = ({config, portInfos}: RobotSetupProps) => {
    const portInfo = portInfos.find((m) => m.serial_id === config.serial_id);
    const connected = portInfo !== undefined;

    return (
        <View>
            <Text>{config.type} robot</Text>
            <Text>{connected ? "connected" : "disconnected"}</Text>
            <Text>Device: {portInfo?.device_name}</Text>
            <Text>Serial: {config.serial_id}</Text>
            <Text>Calibration: {config.id}</Text>
        </View>
    )
}

const useRecordingForm = (datasetName: string | null, project: SchemaProjectConfig) => {
    const isNewDataset = datasetName === null;
    const [dataset, setDataset] = useState<string>(datasetName ?? "");
    const [task, setTask] = useState<string>("");

    return {
        dataset,
        setDataset,
        isNewDataset,
        task,
        setTask,
    }
}

export const Record = () => {
    const { data: availableCameras } = $api.useQuery('get', '/api/hardware/cameras');
    const { data: foundRobots } = $api.useQuery('get', '/api/hardware/robots');
    const [searchParams] = useSearchParams();
    const { project: projectConfig } = useProject();
    const [project, setProject] = useState<SchemaProjectConfig>(projectConfig);
    const { dataset, setDataset, isNewDataset, task, setTask } = useRecordingForm(searchParams.get("dataset"), project);

    const updateCamera = (name: string, id: string, oldId: string) => {
        setProject({
            ...project,
            cameras: project.cameras.map((c) => {
                if (c.name === name) {
                    return { ...c, id }
                } else if (c.id === id) {
                    return { ...c, id: oldId }
                } else {
                    return c;
                }
            })
        });
    }

    return (
        <View flex="1">
            <Heading>Hardware setup</Heading>
            <Form>
                <TextField label='Dataset Name' value={dataset} isDisabled={!isNewDataset} onChange={setDataset} />
                <TextField label='Task' value={task} onChange={setTask} />
            </Form>
            <Tabs>
                <TabList>
                    <Item key="cameras">Cameras</Item>
                    <Item key="robots">Robots</Item>
                </TabList>
                <TabPanels>
                    <Item key="cameras">
                        {project.cameras.map((camera) => <CameraSetup key={camera.name} camera={camera} availableCameras={availableCameras ?? []} updateCamera={updateCamera} />)}
                    </Item>
                    <Item key="robots">
                        {project.robots.map((robot) => <RobotSetup key={robot.serial_id} config={robot} portInfos={foundRobots ?? []}/>)}
                    </Item>

                </TabPanels>

            </Tabs>
        </View>
    );
}