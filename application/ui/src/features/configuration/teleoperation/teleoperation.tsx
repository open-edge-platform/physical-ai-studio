import { Suspense, useState } from 'react';

import {
    Button,
    Text,
    ButtonGroup,
    ComboBox,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    Item,
    Key,
    Loading,
    Section,
    TabList,
    TabPanels,
    Tabs,
    TextField,
    View,
    IllustratedMessage
} from '@geti/ui';

import { ReactComponent as EmptyIllustration } from './../../../assets/illustration.svg';
import { $api } from '../../../api/client';
import { SchemaDatasetOutput, SchemaRobotConfig, SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { useSettings } from '../../../components/settings/use-settings';
import {
    initialTeleoperationConfig,
    makeNameSafeForPath,
} from '../../../routes/datasets/record/utils';
import { useProject } from '../../projects/use-project';
import { CameraSetup } from '../shared/camera-setup';
import { RobotSetup } from '../shared/robot-setup';
import { paths } from '../../../router';

interface TeleoperationSetupProps {
    onDone: (config: SchemaTeleoperationConfig | undefined) => void;
    dataset: SchemaDatasetOutput;
}

export const TeleoperationSetup = ({ dataset, onDone }: TeleoperationSetupProps) => {
    const [activeTab, setActiveTab] = useState<string>('cameras');
    const project = useProject();
    const { data: projectTasks } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/tasks', {
        params: {
            path: { project_id: project.id! },
        },
    });

    const { data: environments } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments', {
        params: {
            path: {
                project_id: project.id
            }
        }
    })
    const { data: environment } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/environments/{environment_id}', {
        params: {
            path: {
                project_id: project.id,
                environment_id: environments[0].id
            }
        }
    });

    const { data: availableCameras, refetch: refreshCameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const { data: foundRobots, refetch: refreshRobots } = $api.useSuspenseQuery('get', '/api/hardware/robots');
    const { data: availableCalibrations, refetch: refreshCalibrations } = $api.useSuspenseQuery(
        'get',
        '/api/hardware/calibrations'
    );

    const { geti_action_dataset_path } = useSettings();

    const initialTask = Object.values(projectTasks).flat()[0];

    const [config, setConfig] = useState<SchemaTeleoperationConfig>(
        {
            task: initialTask,
            dataset,
            environment,
        }
    );

    //const updateCamera = (name: string, id: string, oldId: string, driver: string, oldDriver: string) => {
    //    setConfig({
    //        ...config,
    //        cameras: config.cameras.map((c) => {
    //            if (c.name === name) {
    //                return { ...c, fingerprint: id, driver };
    //            } else if (c.fingerprint === id && c.driver === (driver === 'webcam' ? 'usb_camera' : driver)) {
    //                return { ...c, fingerprint: oldId, driver: oldDriver };
    //            } else {
    //                return c;
    //            }
    //        }),
    //    });
    //};

    const updateRobot = (type: 'leader' | 'follower', robot_config: SchemaRobotConfig) => {
        setConfig((c) => ({
            ...c,
            [type]: robot_config,
        }));
    };

    const updateDataset = (name: string) => {
        setConfig((c) => ({
            ...c,
            dataset: {
                ...c.dataset,
                name,
                path: `${geti_action_dataset_path}/${makeNameSafeForPath(name)}`,
            },
        }));
    };

    const onTabSwitch = (key: Key) => {
        setActiveTab(key.toString());
    };

    const onNext = async () => {
        if (activeTab === 'cameras') {
            setActiveTab('robots');
        } else {
            onDone(config);
        }
    };

    const onBack = () => {
        if (activeTab === 'robots') {
            setActiveTab('cameras');
        } else {
            onDone(undefined);
        }
    };

    const isValid = () => {
        const datasetNameValid = config.dataset !== null;
        const taskValid = config.task !== '';
        return datasetNameValid && taskValid;
    };

    const onRefresh = () => {
        refreshCameras();
        refreshRobots();
        refreshCalibrations();
    };

    if (environments.length === 0) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <Content>Currently there has not been a environment setup yet.</Content>
                    <Heading>No environment set up yet.</Heading>
                    <View margin={'size-100'}>
                        <Button variant='accent' href={paths.project.environments.new({ project_id: project.id })}>Setup environment</Button>
                    </View>
                </IllustratedMessage>
            </Flex>
        )
    }

    return (
        <View>
            <TextField
                validationState={config.dataset.name == '' ? 'invalid' : 'valid'}
                isRequired
                label='Dataset Name'
                width={'100%'}
                value={config.dataset.name}
                isDisabled={true}
                onChange={updateDataset}
            />
            <ComboBox
                validationState={config.task === '' ? 'invalid' : 'valid'}
                isRequired
                width={'100%'}
                label='Task'
                allowsCustomValue
                inputValue={config.task}
                onInputChange={(task) => setConfig((c) => ({ ...c, task }))}
            >
                {Object.keys(projectTasks).map((datasetName) => (
                    <Section key={datasetName} title={datasetName}>
                        {projectTasks[datasetName].map((task) => (
                            <Item key={`${datasetName}-${task}`}>{task}</Item>
                        ))}
                    </Section>
                ))}
            </ComboBox>
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
};

export const TeleoperationSetupModal = (
    close: (config: SchemaTeleoperationConfig | undefined) => void,
    dataset: SchemaDatasetOutput
) => {
    return (
        <Dialog>
            <Heading>Teleoperate Setup</Heading>
            <Divider />
            <Content>
                <Suspense fallback={<Loading mode='overlay' />}>
                    <TeleoperationSetup dataset={dataset} onDone={close} />
                </Suspense>
            </Content>
        </Dialog>
    );
};
