// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Checkbox, Flex, Heading, Text, TextField, toast, View } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { SchemaEnvironmentWithRelations, SchemaSimulationCamera } from '../../api/openapi-spec';
import { useProjectId } from '../projects/use-project';
import { useSimulatedRobotTypes } from './robot-simulation';
import { useRobotSimulation } from './use-robot-simulation';

import classes from './simulation.module.css';

const DISMISSED_PREFIX = 'simulation-cameras-dismissed:';

const isDismissed = (key: string) => {
    try {
        return window.sessionStorage.getItem(DISMISSED_PREFIX + key) !== null;
    } catch {
        return false;
    }
};

const dismiss = (key: string) => {
    try {
        window.sessionStorage.setItem(DISMISSED_PREFIX + key, '1');
    } catch {
        // Only a convenience: the prompt shows again.
    }
};

const SOURCE_LABEL: Record<string, string> = { scene: 'scene', override: 'robot', model: 'robot', default: 'robot' };

const cameraDetails = ({ source, width, height, fps }: SchemaSimulationCamera) =>
    `${SOURCE_LABEL[source] ?? source} camera · ${width}×${height} · ${fps} fps`;

const RobotCamerasPrompt = ({
    environment,
    robot,
}: {
    environment: SchemaEnvironmentWithRelations;
    robot: { id: string; name: string };
}) => {
    const { project_id } = useProjectId();
    const dismissKey = `${environment.id}:${robot.id}`;
    const [isHidden, setIsHidden] = useState(() => isDismissed(dismissKey));
    const { data: simulation } = useRobotSimulation(project_id, robot.id, !isHidden);
    const [selected, setSelected] = useState<Set<string> | undefined>();
    // Only the names the user edited; the rest are named "<robot> <camera>" by the backend.
    const [names, setNames] = useState<Record<string, string>>({});
    const addMutation = $api.useMutation(
        'post',
        '/api/projects/{project_id}/environments/{environment_id}/simulation-cameras',
        {
            meta: {
                // Awaited: the prompt must be gone (or updated) before Add can be pressed again.
                awaits: [
                    [
                        'get',
                        '/api/projects/{project_id}/environments/{environment_id}',
                        { params: { path: { project_id, environment_id: environment.id } } },
                    ],
                    ['get', '/api/projects/{project_id}/cameras', { params: { path: { project_id } } }],
                ],
            },
        }
    );

    // A camera on a stream of this simulation is present when it is this robot's, or no robot's
    // (adding it makes it this robot's). Another robot's camera only shares the URL: that robot's
    // simulation ran on this port before, and moves the camera when it restarts.
    const environmentUrls = new Set(
        (environment.cameras ?? []).flatMap((camera) =>
            camera.driver === 'ipcam' &&
            'url' in camera.payload &&
            [robot.id, null, undefined].includes(camera.payload.simulation_robot_id)
                ? [camera.payload.url]
                : []
        )
    );
    const missing = (simulation?.cameras ?? []).filter(({ stream_url }) => !environmentUrls.has(stream_url));

    if (isHidden || simulation === undefined || !simulation.running || missing.length === 0) {
        return null;
    }

    const chosen = selected ?? new Set(missing.map(({ name }) => name));
    const toggle = (name: string, isSelected: boolean) => {
        const next = new Set(chosen);
        if (isSelected) {
            next.add(name);
        } else {
            next.delete(name);
        }
        setSelected(next);
    };

    const defaultName = (camera: string) => `${robot.name} ${camera}`;
    const add = () => {
        const cameras = missing.filter(({ name }) => chosen.has(name)).map(({ name }) => name);
        addMutation.mutate(
            {
                params: { path: { project_id, environment_id: environment.id } },
                body: {
                    robot_id: robot.id,
                    cameras,
                    names: Object.fromEntries(
                        cameras.flatMap((camera) => (camera in names ? [[camera, names[camera]]] : []))
                    ),
                },
            },
            { onError: (error) => toast.negative(getApiErrorMessage(error) ?? 'Could not add the cameras.') }
        );
    };

    return (
        <View UNSAFE_className={classes.prompt} padding='size-200' marginX='size-200' marginBottom='size-200'>
            <Flex direction='column' gap='size-150'>
                <Heading level={4} margin={0}>
                    Add simulated cameras?
                </Heading>
                <Text>
                    The simulation of {robot.name} streams{' '}
                    {missing.length === 1 ? 'a camera' : `${missing.length} cameras`} that “{environment.name}” does not
                    have yet.
                </Text>
                <Flex gap='size-200' wrap>
                    {missing.map((camera) => (
                        <Flex
                            key={camera.name}
                            direction='column'
                            gap='size-50'
                            UNSAFE_className={classes.cameraChoice}
                        >
                            {/* Through Studio's own address, like the viewer. */}
                            {camera.snapshot_proxy_url != null ? (
                                <img
                                    src={camera.snapshot_proxy_url}
                                    alt={`${camera.name} snapshot`}
                                    className={classes.snapshot}
                                />
                            ) : null}
                            <Checkbox
                                isSelected={chosen.has(camera.name)}
                                onChange={(value) => toggle(camera.name, value)}
                            >
                                {camera.name}
                            </Checkbox>
                            <TextField
                                aria-label={`Name for ${camera.name}`}
                                value={names[camera.name] ?? defaultName(camera.name)}
                                onChange={(value) => setNames((current) => ({ ...current, [camera.name]: value }))}
                                isDisabled={!chosen.has(camera.name)}
                                width='100%'
                            />
                            <Text UNSAFE_className={classes.detail}>{cameraDetails(camera)}</Text>
                        </Flex>
                    ))}
                </Flex>
                <Text UNSAFE_className={classes.detail}>
                    Added as IP cameras on the simulation&apos;s streams, under the names above.
                </Text>
                <ButtonGroup>
                    <Button
                        variant='secondary'
                        onPress={() => {
                            dismiss(dismissKey);
                            setIsHidden(true);
                        }}
                    >
                        Not now
                    </Button>
                    <Button
                        variant='accent'
                        onPress={add}
                        isPending={addMutation.isPending}
                        isDisabled={chosen.size === 0}
                    >
                        {chosen.size === 1 ? 'Add 1 camera' : `Add ${chosen.size} cameras`}
                    </Button>
                </ButtonGroup>
            </Flex>
        </View>
    );
};

/** Offers to add the cameras of the environment's running simulations that it does not have yet. */
export const SimulationCamerasPrompt = ({ environment }: { environment: SchemaEnvironmentWithRelations }) => {
    const simulatedTypes = useSimulatedRobotTypes();
    const robots = (environment.robots ?? []).map(({ robot }) => robot).filter(({ type }) => simulatedTypes.has(type));

    return (
        <>
            {robots.map((robot) => (
                <RobotCamerasPrompt key={robot.id} environment={environment} robot={robot} />
            ))}
        </>
    );
};
