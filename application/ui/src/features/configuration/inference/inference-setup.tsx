import { Suspense, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    Item,
    Key,
    Loading,
    TabList,
    TabPanels,
    Tabs,
    View,
} from '@geti/ui';


import { SchemaInferenceConfig, SchemaTeleoperationConfig } from "../../../api/openapi-spec";
import { TELEOPERATION_CONFIG_CACHE_KEY } from '../../../routes/datasets/record/utils';
import { $api } from '../../../api/client';
import { useProject } from '../../projects/use-project';
import { CameraSetup } from '../shared/camera-setup';
import { RobotSetup } from '../shared/robot-setup';

interface InferenceSetupProps {
  onDone: (config: SchemaInferenceConfig | undefined) => void;
    model_id: string | undefined;
}

export const InferenceSetup = ({model_id, onDone}: InferenceSetupProps) => {
    const [activeTab, setActiveTab] = useState<string>('cameras');
    const project = useProject();
    const { data: model } = $api.useSuspenseQuery("get", "/api/models/{model_id}", { params: { query: { uuid: model_id } } })
    const { data: availableCameras, refetch: refreshCameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const { data: foundRobots, refetch: refreshRobots } = $api.useSuspenseQuery('get', '/api/hardware/robots');
    const { data: availableCalibrations, refetch: refreshCalibrations } = $api.useSuspenseQuery(
        'get',
        '/api/hardware/calibrations'
    );

    const createDatasetMutation = $api.useMutation('post', '/api/dataset');

    const cachedConfig = JSON.parse(localStorage.getItem(TELEOPERATION_CONFIG_CACHE_KEY) ?? "{}") as SchemaTeleoperationConfig

    // TODO: make cached config better...
    const [config, setConfig] = useState<SchemaInferenceConfig>(
      {
        model,
        task_index: 0,
        fps: project.config!.fps,
        cameras: cachedConfig.cameras,
        robot: cachedConfig.follower,
      }
    );

    const updateCamera = (name: string, id: string, oldId: string, driver: string, oldDriver: string) => {
        setConfig({
            ...config,
            cameras: config.cameras.map((c) => {
                if (c.name === name) {
                    return { ...c, port_or_device_id: id, driver };
                } else if (c.port_or_device_id === id && c.driver === driver) {
                    return { ...c, port_or_device_id: oldId, driver: oldDriver };
                } else {
                    return c;
                }
            }),
        });
    };

    const onTabSwitch = (key: Key) => {
        setActiveTab(key.toString());
    };

    const onNext = async () => {
        if (activeTab === 'cameras') {
            setActiveTab('robots');
        } else {
            //storeConfigToCache(config);
            onDone(config);
        }
    };

    const isValid = () => {
      return true;
    }

    const onBack = () => {
        if (activeTab === 'robots') {
            setActiveTab('cameras');
        } else {
            onDone(undefined);
        }
    };

    const onRefresh = () => {
        refreshCameras();
        refreshRobots();
        refreshCalibrations();
    };

    return (
        <View>
            <View height={'330px'}>
                <Tabs onSelectionChange={onTabSwitch} selectedKey={activeTab}>
                    <TabList>
                        <Item key='cameras'>Cameras</Item>
                        <Item key='robots'>Robots</Item>
                    </TabList>
                    <TabPanels>
                        <Item key='cameras'>
                            <Flex gap='40px'>
                                {config.cameras.map((camera) => (
                                    <CameraSetup
                                        key={camera.name}
                                        camera={camera}
                                        availableCameras={availableCameras ?? []}
                                        updateCamera={updateCamera}
                                    />
                                ))}
                            </Flex>
                        </Item>
                        <Item key='robots'>
                            <Flex gap='40px'>
                                <RobotSetup
                                    key={'robot'}
                                    config={config.robot}
                                    portInfos={foundRobots ?? []}
                                    calibrations={availableCalibrations ?? []}
                                    setConfig={(robot) => setConfig((r) => ({...r, robot}))}
                                />
                            </Flex>
                        </Item>
                    </TabPanels>
                </Tabs>
            </View>
            <Flex justifyContent={'end'}>
                <View paddingTop={'size-300'}>
                    <ButtonGroup>
                        <Button onPress={onRefresh} variant='secondary'>
                            Refresh
                        </Button>
                        <Button onPress={onBack} variant='secondary'>
                            {activeTab === 'robots' ? 'Back' : 'Cancel'}
                        </Button>
                        <Button onPress={onNext} isDisabled={activeTab == 'robots' && !isValid()}>
                            {activeTab === 'robots' ? 'Start' : 'Next'}
                        </Button>
                    </ButtonGroup>
                </View>
            </Flex>
        </View>
    );
}

export const InferenceSetupModal = (
    close: (config: SchemaInferenceConfig | undefined) => void,
    model_id: string | undefined
) => {
    return (
        <Dialog>
            <Heading>Inference Setup</Heading>
            <Divider />
            <Content>
                <Suspense fallback={<Loading mode='overlay' />}>
                    <InferenceSetup model_id={model_id} onDone={close} />
                </Suspense>
            </Content>
        </Dialog>
    );
};
