// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import { Button, Flex, Link, ProgressCircle, StatusLight, Text, toast, View } from '@geti-ui/ui';

import { getApiErrorMessage } from '../../api/errors';
import { useProjectId } from '../projects/use-project';
import { useSimulationDialog } from './simulation-dialog';
import { SimulationNotRunning } from './simulation-not-running';
import { RobotSimulation, useRobotSimulation } from './use-robot-simulation';
import { useSimulation, useStopSimulation } from './use-simulation';

import classes from './simulation.module.css';

/**
 * The simulation's own 3D viewer (viser) with a thin status bar, in place of the URDF view of a
 * simulated robot. Shows how to start the simulation while it does not run.
 */
export const SimulationViewer = ({
    robot,
    actions,
    children,
    compact = false,
}: {
    robot: { id: string; name: string; type: string };
    /** Controls of the surrounding view, shown at the end of the bar. */
    actions?: ReactNode;
    /** Shown below the viewer while the simulation runs. */
    children?: ReactNode;
    /** A smaller bar, for panels. */
    compact?: boolean;
}) => {
    const { project_id } = useProjectId();
    const { data: simulation, isPending } = useRobotSimulation(project_id, robot.id);
    const { data: studioSimulation } = useSimulation();
    // The dialog is rendered in every state below: it has to outlive the "not running" view.
    const simulationDialog = useSimulationDialog(robot);

    let content: ReactNode;
    if (isPending) {
        content = (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <ProgressCircle isIndeterminate aria-label='Looking for the simulation' />
            </Flex>
        );
    } else if (simulation === undefined || !simulation.running) {
        const isStarting =
            studioSimulation?.robot_id === robot.id &&
            (studioSimulation.status === 'starting' || studioSimulation.status === 'stopping');
        content = <SimulationNotRunning robot={robot} isStarting={isStarting} onStart={simulationDialog.open} />;
    } else {
        content = (
            <RunningSimulation robot={robot} simulation={simulation} actions={actions} compact={compact}>
                {children}
            </RunningSimulation>
        );
    }

    return (
        <>
            {content}
            {simulationDialog.dialog}
        </>
    );
};

const RunningSimulation = ({
    robot,
    simulation,
    actions,
    children,
    compact,
}: {
    robot: { name: string };
    simulation: RobotSimulation;
    actions?: ReactNode;
    children?: ReactNode;
    compact: boolean;
}) => {
    const { project_id } = useProjectId();
    const stopMutation = useStopSimulation(project_id);

    const startedByStudio = simulation.started_by === 'studio';

    return (
        <Flex direction='column' width='100%' height='100%'>
            <View UNSAFE_className={classes.viewerBar} paddingX='size-150' paddingY='size-75'>
                <Flex alignItems='center' gap='size-150' wrap>
                    <StatusLight variant='positive' UNSAFE_className={classes.viewerStatus}>
                        Simulation running
                    </StatusLight>
                    {compact || startedByStudio ? null : (
                        <Text UNSAFE_className={classes.detail}>started outside Studio</Text>
                    )}
                    {simulation.scene != null ? (
                        <Text UNSAFE_className={classes.detail}>{simulation.scene}</Text>
                    ) : null}
                    <Flex marginStart='auto' gap='size-100' alignItems='center'>
                        {simulation.viewer_proxy_url != null ? (
                            <Link href={simulation.viewer_proxy_url} target='_blank' rel='noreferrer' isQuiet>
                                Open viewer ↗
                            </Link>
                        ) : null}
                        {startedByStudio && !compact ? (
                            <Button
                                variant='secondary'
                                isPending={stopMutation.isPending}
                                onPress={() =>
                                    stopMutation.mutate(
                                        {},
                                        {
                                            onError: (error) =>
                                                toast.negative(getApiErrorMessage(error) ?? 'Failed to stop.'),
                                        }
                                    )
                                }
                            >
                                Stop simulation
                            </Button>
                        ) : null}
                        {actions}
                    </Flex>
                </Flex>
            </View>
            {/* Through Studio's own address: the simulation's ports may be out of the browser's reach (Docker). */}
            {simulation.viewer_proxy_url != null ? (
                <iframe
                    className={classes.viewer}
                    src={simulation.viewer_proxy_url}
                    title={`${robot.name} simulation viewer`}
                    allow='fullscreen'
                />
            ) : (
                <Flex flex={1} justifyContent='center' alignItems='center'>
                    <Text>The simulation runs without a viewer.</Text>
                </Flex>
            )}
            {children}
        </Flex>
    );
};
