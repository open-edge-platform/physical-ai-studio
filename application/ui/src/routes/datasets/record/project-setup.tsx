import { useState } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Flex,
    Heading,
    Item,
    NumberField,
    TabList,
    TabPanels,
    Tabs,
    TextField,
    View,
    Well,
} from '@geti/ui';
import { Close } from '@geti/ui/icons';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import { SchemaCameraConfigInput, SchemaProjectConfigInput } from '../../../api/openapi-spec';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';

interface CameraSetupProps {
    camera: SchemaCameraConfigInput;
    updateCamera: (camera: SchemaCameraConfigInput) => void;
    removeCamera: () => void;
}
const CameraSetup = ({ camera, updateCamera, removeCamera }: CameraSetupProps) => {
    return (
        <Well>
            <Flex justifyContent={'end'}>
                <ActionButton aria-label='Remove Camera' onPress={removeCamera}>
                    <Close />
                </ActionButton>
            </Flex>
            <Flex direction={'column'}>
                <TextField
                    label='Name'
                    value={camera.name}
                    isRequired
                    onChange={(name) => updateCamera({ ...camera, name })}
                />
                <NumberField
                    label='FPS'
                    value={camera.fps}
                    minValue={30}
                    onChange={(fps) => updateCamera({ ...camera, fps })}
                />
                <NumberField
                    label='Width'
                    value={camera.width}
                    onChange={(width) => updateCamera({ ...camera, width })}
                />
                <NumberField
                    label='Height'
                    value={camera.height}
                    onChange={(height) => updateCamera({ ...camera, height })}
                />
            </Flex>
        </Well>
    );
};

const emptyCamera = (): SchemaCameraConfigInput => {
    return {
        driver: 'webcam',
        fps: 30,
        width: 640,
        height: 480,
        id: uuidv4(),
        name: '',
        port_or_device_id: '',
        use_depth: false,
    };
};

export const ProjectSetup = () => {
    const navigate = useNavigate();

    const { project_id } = useProjectId();
    const [activeTab, setActiveTab] = useState<string>('robot');
    const [config, setConfig] = useState<SchemaProjectConfigInput>({
        id: uuidv4(),
        fps: 30,
        cameras: [emptyCamera()],
        robot_type: '',
    });

    const saveMutation = $api.useMutation('put', '/api/projects/{project_id}/project_config', {});

    const selectRobot = (robot_type: SchemaProjectConfigInput['robot_type']) => {
        setConfig((c) => ({ ...c, robot_type }));
        setActiveTab('cameras');
    };

    const addCamera = () => {
        setConfig((c) => ({ ...c, cameras: [...c.cameras, emptyCamera()] }));
    };

    const removeCamera = (id: string) => {
        setConfig((c) => ({ ...c, cameras: c.cameras.filter((b) => b.id !== id) }));
    };

    const updateCamera = (camera: SchemaCameraConfigInput) => {
        setConfig((c) => ({
            ...c,
            cameras: c.cameras.map((b) => {
                return camera.id === b.id ? camera : b;
            }),
        }));
    };

    const onSave = () => {
        saveMutation.mutate({
            params: {
                path: { project_id },
            },
            body: config,
        });
    };

    const onBack = () => {
        if (activeTab === 'cameras') {
            setActiveTab('robot');
        } else {
            navigate(paths.project.datasets.index({ project_id }));
        }
    };

    const canSave = () => {
        return (
            config.robot_type !== '' &&
            config.cameras.length > 0 &&
            config.cameras.find((c) => c.name === '') === undefined
        );
    };

    return (
        <Flex justifyContent={'center'} flex='1'>
            <View flex='1' padding='size-150' maxWidth={'640px'}>
                <View paddingTop={'size-100'} paddingBottom={'size-100'}>
                    <Heading>Project Configuration</Heading>
                </View>
                <View backgroundColor={'gray-200'} padding={'size-200'}>
                    <Tabs onSelectionChange={(key) => setActiveTab(key.toString())} selectedKey={activeTab}>
                        <TabList>
                            <Item key='robot'>Robot</Item>
                            <Item key='cameras'>Cameras</Item>
                        </TabList>
                        <TabPanels>
                            <Item key='robot'>
                                <Content justifySelf={'center'} margin={'size-100'}>
                                    Choose a robot
                                </Content>
                                <Flex height='size-2000' gap='size-200' justifyContent={'center'}>
                                    <Button
                                        variant={config.robot_type === 'so101_follower' ? 'primary' : 'secondary'}
                                        width='size-2000'
                                        height='size-2000'
                                        onPress={() => selectRobot('so101_follower')}
                                    >
                                        SO 101
                                    </Button>
                                    <Button variant='secondary' width='size-2000' height='size-2000'>
                                        More coming soon
                                    </Button>
                                </Flex>
                            </Item>
                            <Item key='cameras'>
                                <Flex direction={'column'}>
                                    <Content alignSelf={'center'} margin={'size-100'}>
                                        Choose a robot
                                    </Content>
                                    <Button alignSelf={'end'} onPress={addCamera}>
                                        Add camera
                                    </Button>
                                    <Flex wrap gap='size-200'>
                                        {config.cameras.map((camera) => (
                                            <CameraSetup
                                                key={camera.id}
                                                camera={camera}
                                                removeCamera={() => camera.id && removeCamera(camera.id)}
                                                updateCamera={updateCamera}
                                            />
                                        ))}
                                    </Flex>
                                </Flex>
                            </Item>
                        </TabPanels>
                    </Tabs>
                    <Flex justifyContent={'end'} margin={'size-100'}>
                        <ButtonGroup>
                            <Button variant='secondary' onPress={onBack}>
                                Back
                            </Button>
                            <Button isDisabled={saveMutation.isPending || !canSave()} onPress={onSave}>
                                Save
                            </Button>
                        </ButtonGroup>
                    </Flex>
                </View>
            </View>
        </Flex>
    );
};
