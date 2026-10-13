// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Flex, View } from '@geti-ui/ui';

import { useSimulationDialog } from './simulation-dialog';
import { SimulationNotRunning } from './simulation-not-running';
import { useSimulation } from './use-simulation';

/**
 * Shown by a recording or inference session while its follower's simulation does not run: the
 * session connects by itself once the simulation started here (or elsewhere) runs.
 */
export const SessionSimulationNotRunning = ({
    robot,
    children,
}: {
    robot: { id: string; name: string; type: string };
    /** Shown below, e.g. the session's Cancel button. */
    children?: ReactNode;
}) => {
    const { data: studioSimulation } = useSimulation();
    const simulationDialog = useSimulationDialog(robot);
    const isStarting =
        studioSimulation?.robot_id === robot.id &&
        (studioSimulation.status === 'starting' || studioSimulation.status === 'stopping');

    return (
        <Flex direction='column' width='100%' height='100%'>
            <View flex={1}>
                <SimulationNotRunning robot={robot} isStarting={isStarting} onStart={simulationDialog.open} />
            </View>
            <Flex justifyContent='center' marginBottom='size-300'>
                {children}
            </Flex>
            {simulationDialog.dialog}
        </Flex>
    );
};
