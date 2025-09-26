import { Flex, Key, Heading, Text, View, Picker, Item, Button } from '@geti/ui';

import { SchemaCalibrationConfig, SchemaRobotConfig, SchemaRobotPortInfo } from '../../../api/openapi-spec';
import { $api } from '../../../api/client';

const ConnectionIcon = ({ radius, color }: { radius: number; color: string }) => {
    return (
        <svg fill={color} width={radius * 2} height={radius * 2}>
            <circle cx={radius} cy={radius} r={radius} />
        </svg>
    );
};

interface RobotSetupProps {
    config: SchemaRobotConfig;
    portInfos: SchemaRobotPortInfo[];
    calibrations: SchemaCalibrationConfig[];
    setConfig: (config: SchemaRobotConfig) => void;
}

const matchRobotType = (portInfo: SchemaRobotPortInfo, config: SchemaRobotConfig): boolean => {
    if (config.robot_type === "")
        return true;
    if (config.robot_type === "so101_follower") {
        if (portInfo.robot_type === "so-100") {
            return true;
        }
    }
    return false;
}

export const RobotSetup = ({ config, portInfos, calibrations, setConfig }: RobotSetupProps) => {
    const portInfo = portInfos.find((m) => m.serial_id === config.serial_id);
    const connected = portInfo !== undefined;

    const identifyMutation = $api.useMutation('put', '/api/hardware/identify');

    const serialIdOptions = portInfos.filter((portInfo) => matchRobotType(portInfo, config)).map((r) => ({ id: r.serial_id, name: r.serial_id }));
    const calibrationOptions = calibrations.filter((c) => c.robot_type === config.type).map((r) => ({ id: r.id, name: r.id }));

    const selectRobot = (id: Key | null) => {
        setConfig({...config, serial_id: id?.toString() ?? ""})
    }

    const selectCalibration = (id: Key | null) => {
        setConfig({...config, id: id?.toString() ?? ""})
    }

    const identify = () => {
        const portInfo = portInfos.find((m) => m.serial_id === config.serial_id);

        if (portInfo) {
            identifyMutation.mutate({
                body: portInfo,
            });
        }
    };

    return (
        <Flex flex='1'>
            <View backgroundColor={'gray-100'} flex='1' padding={'size-200'} marginTop={'size-100'}>
                <Flex direction={'column'} justifyContent={'space-between'} gap="size-100">
                    <Flex justifyContent={'space-between'}>
                        <Heading>{config.type} robot</Heading>
                        <Flex justifyContent={'center'} alignItems={'center'}>
                            <ConnectionIcon radius={3} color={connected ? 'green' : 'red'} />
                            <Text UNSAFE_style={{ marginLeft: '5px' }}>
                                {connected ? 'connected' : 'disconnected'}
                            </Text>
                        </Flex>
                    </Flex>
                    <Picker
                        items={serialIdOptions}
                        selectedKey={config.serial_id}
                        label='Serial ID'
                        onSelectionChange={selectRobot}
                        flex={1}
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>
                    <Picker
                        items={calibrationOptions}
                        label='Calibration'
                        selectedKey={config.id}
                        onSelectionChange={selectCalibration}
                    >
                        {(item) => <Item>{item.name}</Item>}
                    </Picker>
                    <Button alignSelf={'end'} isDisabled={config.serial_id === ''} onPress={identify}>
                        Identify
                    </Button>
                </Flex>
            </View>
        </Flex>
    );
};
