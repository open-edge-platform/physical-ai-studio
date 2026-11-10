import { ReactNode } from 'react';

import { Badge, Flex, Text, View } from '@geti-ui/ui';

import { DeviceStatus } from './session-devices';

import classes from './devices.module.css';

interface DeviceCardProps {
    thumb?: ReactNode;
    title: string;
    subtitle: string;
    status: DeviceStatus;
    error?: string | null;
}

export const DeviceCard = ({ thumb, title, subtitle, status, error }: DeviceCardProps) => (
    <View
        backgroundColor='gray-75'
        borderWidth='thin'
        borderColor='gray-300'
        borderRadius='small'
        paddingY='size-100'
        paddingX='size-150'
        minWidth={0}
    >
        <Flex direction='column' gap='size-50'>
            <Flex alignItems='center' gap='size-150'>
                {thumb}
                <Flex direction='column' flexGrow={1} minWidth={0}>
                    <Text UNSAFE_className={classes.deviceName}>{title}</Text>
                    <Text UNSAFE_className={classes.deviceSubtitle}>{subtitle}</Text>
                </Flex>
                <Badge variant={status.variant} flexShrink={0}>
                    {status.label}
                </Badge>
            </Flex>
            {error && (
                <details className={classes.deviceError}>
                    <summary>Error details</summary>
                    <pre className={classes.deviceErrorMessage}>{error}</pre>
                </details>
            )}
        </Flex>
    </View>
);
