// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
}: {
    robot: { name: string };
    onStart?: () => void;
    /** Connects again, for after the simulation was started outside Studio. */
    onReconnect?: () => void;
    /** Studio is starting (or stopping) the simulation: offer its progress instead. */
    isStarting?: boolean;
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
                    {isStarting ? 'Simulation starting' : 'Simulation not running'}
                </Heading>
                <Text>
                    {robot.name} needs a running simulation. Start it to see it here, move its joints and use it in
                    environments.
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
                </Flex>
            </Flex>
        </View>
    );
};
