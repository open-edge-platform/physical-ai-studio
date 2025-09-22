import { Flex, Heading, Text, View } from '@geti/ui';

import { SchemaRobotConfig, SchemaRobotPortInfo } from '../../../api/openapi-spec';

interface RobotSetupProps {
    config: SchemaRobotConfig;
    portInfos: SchemaRobotPortInfo[];
}

const ConnectionIcon = ({ radius, color }: { radius: number; color: string }) => {
    return (
        <svg fill={color} width={radius * 2} height={radius * 2}>
            <circle cx={radius} cy={radius} r={radius} />
        </svg>
    );
};
export const RobotSetup = ({ config, portInfos }: RobotSetupProps) => {
    const portInfo = portInfos.find((m) => m.serial_id === config.serial_id);
    const connected = portInfo !== undefined;

    return (
        <Flex flex='1'>
            <View backgroundColor={'gray-100'} flex='1' padding={'size-200'} marginTop={'size-100'}>
                <Flex direction={'column'} justifyContent={'space-between'} height='130px'>
                    <View marginBottom={'size-100'}>
                        <Flex justifyContent={'space-between'}>
                            <Heading>{config.type} robot</Heading>
                            <Flex justifyContent={'center'} alignItems={'center'}>
                                <ConnectionIcon radius={3} color={connected ? 'green' : 'red'} />
                                <Text UNSAFE_style={{ marginLeft: '5px' }}>
                                    {connected ? 'connected' : 'disconnected'}
                                </Text>
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
    );
};
