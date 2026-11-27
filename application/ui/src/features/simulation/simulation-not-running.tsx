// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import { Button, Flex, Heading, Text, View } from '@geti-ui/ui';

/**
 * Shown where a simulated robot should appear while its simulation is not running: offers to start
 * it (`onStart`, from `useSimulationDialog`), or to connect again after it was started elsewhere.
 */
export const SimulationNotRunning = ({
    robot,
    onStart,
    onReconnect,
    isStarting = false,
    title,
    description,
    actions,
}: {
    robot: { name: string };
    onStart?: () => void;
    /** Connects again, for after the simulation was started outside Studio. */
    onReconnect?: () => void;
    /** Studio is starting (or stopping) the simulation: offer its progress instead. */
    isStarting?: boolean;
    /** In place of the heading the simulation's state gives. */
    title?: string;
    /** What the simulation is needed for, in place of the default. */
    description?: ReactNode;
    /** More buttons, after the start button. */
    actions?: ReactNode;
}) => {
    return (
        <View width='100%' height='100%' padding='size-300'>
            <Flex
                width='100%'
                height='100%'
                direction='column'
                justifyContent='center'
                alignItems='center'
                gap='size-200'
            >
                <Heading level={3} margin={0}>
                    {title ?? (isStarting ? 'Simulation starting' : 'Simulation not running')}
                </Heading>
                <Text>
                    {description ??
                        `${robot.name} needs a running simulation. ` +
                            'Start it to see it here, move its joints and use it in environments.'}
                </Text>
                <Flex gap='size-100'>
                    {onStart !== undefined ? (
                        <Button variant='accent' onPress={onStart}>
                            {isStarting ? 'Show progress…' : 'Start simulation…'}
                        </Button>
                    ) : null}
                    {onReconnect !== undefined ? (
                        <Button variant='secondary' onPress={onReconnect}>
                            Reconnect
                        </Button>
                    ) : null}
                    {actions}
                </Flex>
            </Flex>
        </View>
    );
};
