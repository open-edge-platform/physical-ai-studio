// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { Flex, Text } from '@geti-ui/ui';

import { SchemaSimulationCameraInput } from '../../api/openapi-spec';
import { useProjectId } from '../projects/use-project';
import { useRobotSimulation } from '../simulation/use-robot-simulation';

export const RETRY_STREAM_MS = 3_000;

/**
 * The camera's stream through Studio's own address, like the simulation's viewer.
 *
 * Opened again whenever the simulation starts somewhere else (a stream can end without an error),
 * and retried while it fails.
 */
export const SimulationCameraStream = ({ camera }: { camera: SchemaSimulationCameraInput }) => {
    const { project_id } = useProjectId();
    const [attempt, setAttempt] = useState(0);
    const [failed, setFailed] = useState(false);
    const { robot_id, camera: name } = camera.payload;
    const { data: simulation } = useRobotSimulation(project_id, robot_id);
    const isStopped = simulation !== undefined && !simulation.running;
    const base = `/api/projects/${project_id}/robots/${robot_id}/simulation`;
    const stream = `${base}/http/cameras/${encodeURIComponent(name)}/mjpeg`;

    useEffect(() => {
        if (!failed) {
            return;
        }
        // The simulation may start, or restart elsewhere, at any time: try the stream again.
        const timer = setTimeout(() => {
            setFailed(false);
            setAttempt((current) => current + 1);
        }, RETRY_STREAM_MS);

        return () => clearTimeout(timer);
    }, [failed]);

    return (
        <>
            {(failed || isStopped) && (
                <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                    <Text>Start the simulation to see this camera.</Text>
                </Flex>
            )}
            {!isStopped && (
                <img
                    // A new address is a new simulation: open its stream.
                    key={`${simulation?.http_url ?? ''}-${attempt}`}
                    src={attempt === 0 ? stream : `${stream}?attempt=${attempt}`}
                    alt={`Camera: ${camera.name}`}
                    onError={() => setFailed(true)}
                    style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        display: failed ? 'none' : undefined,
                    }}
                />
            )}
        </>
    );
};
