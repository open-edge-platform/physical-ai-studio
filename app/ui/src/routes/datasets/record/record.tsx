import { View, Form, Heading, TextField, Picker, Section, ActionButton, Text, Flex, Key, Well, Tabs, TabList, TabPanels, Item, Button, ButtonGroup } from '@geti/ui'
import { useProject } from '../../projects/project.provider';
import { useLocation, useNavigate, useParams } from 'react-router';
import { useSearchParams } from 'react-router-dom';
import { useState } from 'react';
import { SchemaCameraConfig, SchemaCamera, SchemaProjectConfig, SchemaRobotConfig, SchemaRobotPortInfo } from '../../../api/openapi-spec';
import { $api } from '../../../api/client';
import { paths } from '../../../router';


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
        <Flex direction={'column'} flex={1}>
            <Heading>{camera.name}</Heading>
            <img src={`/api/hardware/camera_feed?id=${camera.id}&type=${camera.type}`} style={{ flex: 1, maxWidth: "280px", paddingBottom: "10px" }} />
            <Picker selectedKey={camera.id} onSelectionChange={onSelection}>
                {camerasConnectedOfType.map((camera) => (
                    <Item key={camera.id}>{camera.name}</Item>
                ))}
            </Picker>
        </Flex>
    );
}

interface RobotSetupProps {
    config: SchemaRobotConfig,
    portInfos: SchemaRobotPortInfo[]
}

const ConnectionIcon = ({ radius, color }: { radius: number, color: string }) => {
    return (
        <svg fill={color} width={radius * 2} height={radius * 2}>
            <circle cx={radius} cy={radius} r={radius} />
        </svg>
    );
}
const RobotSetup = ({ config, portInfos }: RobotSetupProps) => {
    const portInfo = portInfos.find((m) => m.serial_id === config.serial_id);
    const connected = portInfo !== undefined;

    return (
        <Flex flex="1">
            <View backgroundColor={"gray-100"} flex="1" padding={"size-200"} marginTop={"size-100"}>
                <Flex direction={'column'} justifyContent={'space-between'} height="130px" >
                    <View marginBottom={"size-100"}>
                        <Flex justifyContent={"space-between"}>
                            <Heading>{config.type} robot</Heading>
                            <Flex justifyContent={"center"} alignItems={"center"}>
                                <ConnectionIcon radius={3} color={connected ? "green" : "red"} />
                                <Text UNSAFE_style={{ marginLeft: "5px" }}>{connected ? "connected" : "disconnected"}</Text>
                            </Flex>
                        </Flex>
                    </View>
                    <Flex direction={'column'}>
                        <Text>Device: {portInfo?.device_name}</Text>
                        <Text>Serial: {config.serial_id}</Text>
                        <Text>Calibration: {config.id}</Text>
                    </Flex>
                </Flex>
            </View>
        </Flex>
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


    const navigate = useNavigate();
    console.log(project);

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

    const [activeTab, setActiveTab] = useState<string>("cameras")

    const onTabSwitch = (key: Key) => {
        setActiveTab(key.toString());
    }

    const onNext = () => {
        if (activeTab === "cameras") {
            setActiveTab("robots");
        } else {
            //lets start!
        }
    }

    const onBack = () => {
        if (activeTab === "robots") {
            setActiveTab("cameras");
        } else {
            navigate(paths.project.datasets.index({ project_id: project.id }));
        }
    }

    const isValid = () => {
        const datasetNameValid = dataset !== ""
        const taskValid = task !== ""
        return datasetNameValid && taskValid
    }

    return (
        <Flex justifyContent={"center"} flex="1">
            <View flex="1" padding="size-150" maxWidth={"640px"}>
                <View paddingTop={"size-100"} paddingBottom={"size-100"}>
                    <Heading>Hardware setup</Heading>
                </View>
                <View backgroundColor={'gray-200'} padding={"size-200"}>
                    <Form>
                        <TextField validationState={dataset === "" ? "invalid" : "valid"} isRequired label='Dataset Name' value={dataset} isDisabled={!isNewDataset} onChange={setDataset} />
                        <TextField validationState={task === "" ? "invalid" : "valid"} isRequired label='Task' value={task} onChange={setTask} />
                    </Form>
                    <View height={"330px"}>
                        <Tabs onSelectionChange={onTabSwitch} selectedKey={activeTab}>
                            <TabList>
                                <Item key="cameras">Cameras</Item>
                                <Item key="robots">Robots</Item>
                            </TabList>
                            <TabPanels>
                                <Item key="cameras">
                                    <Flex gap="40px">
                                        {project.cameras.map((camera) => <CameraSetup key={camera.name} camera={camera} availableCameras={availableCameras ?? []} updateCamera={updateCamera} />)}
                                    </Flex>
                                </Item>
                                <Item key="robots">
                                    <Flex gap="40px">
                                        {project.robots.map((robot) => <RobotSetup key={robot.serial_id} config={robot} portInfos={foundRobots ?? []} />)}
                                    </Flex>
                                </Item>
                            </TabPanels>
                        </Tabs>
                    </View>
                    <Flex justifyContent={'end'}>
                        <View paddingTop={"size-300"}>
                            <ButtonGroup>
                                <Button onPress={onBack} variant='secondary'>{activeTab === "robots" ? "Back" : "Cancel"}</Button>
                                <Button 
                                    onPress={onNext}
                                    isDisabled={activeTab == "robots" && !isValid()} >
                                    {activeTab === "robots" ? "Start" : "Next"}
                                    </Button>
                            </ButtonGroup>
                        </View>
                    </Flex>
                </View>
            </View>
        </Flex>
    );
}