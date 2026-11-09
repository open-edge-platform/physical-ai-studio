import { ReactNode, Suspense } from 'react';

import { Flex, Loading, Text } from '@geti-ui/ui';

import { CamerasList } from './cameras-list';
import { DeviceCard } from './device-card';
import { RobotsList } from './robots-list';
import { getGenericStatus, SessionDevice } from './session-devices';

import classes from './devices.module.css';

const Section = ({ label, children }: { label: string; children: ReactNode }) => (
    <Flex direction='column' gap='size-100' minWidth={0}>
        <Text UNSAFE_className={classes.sectionLabel}>{label}</Text>
        {children}
    </Flex>
);

export const DeviceCards = ({ devices }: { devices: SessionDevice[] }) => {
    const robots = devices.filter((device) => device.kind === 'robot');
    const cameras = devices.filter((device) => device.kind === 'camera');
    const others = devices.filter((device) => device.kind === 'unknown');

    return (
        <Suspense fallback={<Loading mode='inline' />}>
            <Flex direction='column' gap='size-150'>
                {robots.length > 0 && (
                    <Section label='Robots'>
                        <RobotsList robots={robots} />
                    </Section>
                )}
                {cameras.length > 0 && (
                    <Section label='Cameras'>
                        <CamerasList cameras={cameras} />
                    </Section>
                )}
                {others.length > 0 && (
                    <Section label='Other'>
                        {others.map(({ name, status }) => (
                            <DeviceCard
                                key={name}
                                title={name}
                                subtitle='Device'
                                status={getGenericStatus(status)}
                                error={status.error}
                            />
                        ))}
                    </Section>
                )}
            </Flex>
        </Suspense>
    );
};
