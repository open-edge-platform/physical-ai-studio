// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { keepPreviousData } from '@tanstack/react-query';

import { $api } from '../../api/client';
import { SchemaRobotSimulation } from '../../api/openapi-spec';

export type RobotSimulation = SchemaRobotSimulation;

/** Probes of a simulation started elsewhere that may fail in a row before it counts as stopped. */
const MISSED_PROBES = 2;

/**
 * What a simulated robot's simulation serves (viewer, speed, cameras), polled while shown.
 *
 * A simulation started outside Studio is only known from probes, and a busy one can miss one: it
 * counts as stopped after `MISSED_PROBES` misses in a row, so the viewer does not reload. Studio's
 * own simulation is reported by Studio itself and changes at once.
 */
export const useRobotSimulation = (project_id: string, robot_id: string, enabled = true) => {
    const query = $api.useQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}/simulation',
        { params: { path: { project_id, robot_id } } },
        // A new object per probe, even an identical one: each probe counts below.
        { enabled, refetchInterval: enabled ? 2_000 : false, structuralSharing: false }
    );
    // Keyed by robot: another robot's last result must not stand in for this one's.
    const [kept, setKept] = useState<{ robot_id: string; data: RobotSimulation } | undefined>(undefined);
    const stable = kept?.robot_id === robot_id ? kept.data : undefined;
    const misses = useRef(0);

    useEffect(() => {
        const latest = query.data;
        if (latest === undefined) {
            return;
        }
        if (!latest.running && stable?.running === true && stable.started_by === 'external') {
            misses.current += 1;
            if (misses.current < MISSED_PROBES) {
                return;
            }
        }
        misses.current = 0;
        setKept({ robot_id, data: latest });
    }, [query.data, stable, robot_id]);

    return { ...query, data: stable ?? query.data };
};

/**
 * The command that starts a robot's simulation in a terminal, outside Studio, for a scene and seed
 * (null for random layouts). No data when the robot type's plugin offers none. Fetched again on every
 * open: it carries the robot's address, which Studio's own simulation may have moved.
 */
export const useSimulationCommand = (project_id: string, robot_id: string, scene: string, seed: number | null) =>
    $api.useQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}/simulation/command',
        { params: { path: { project_id, robot_id }, query: seed === null ? { scene } : { scene, seed } } },
        // The last command stays while the next scene's or seed's loads, so the section does not flicker.
        { placeholderData: keepPreviousData, staleTime: 0 }
    );
