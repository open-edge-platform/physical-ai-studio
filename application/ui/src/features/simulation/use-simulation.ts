// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { useQueryClient } from '@tanstack/react-query';

import { $api, fetchClient } from '../../api/client';
import { SchemaSimulation } from '../../api/openapi-spec';

export type Simulation = SchemaSimulation;

/** A simulation Studio started that is still starting, running or stopping. */
export const isActiveSimulation = (simulation: Simulation | undefined): simulation is Simulation =>
    simulation !== undefined &&
    (simulation.status === 'starting' || simulation.status === 'running' || simulation.status === 'stopping');

const simulationQueryKey = () => $api.queryOptions('get', '/api/simulation').queryKey;

/**
 * The simulation Studio started, if any (there is at most one). `useSimulationUpdates` keeps it
 * current; the slow refetch is only a fallback for a dropped event stream.
 */
export const useSimulation = () => {
    return $api.useQuery('get', '/api/simulation', undefined, {
        refetchInterval: (query) => (isActiveSimulation(query.state.data) ? 5_000 : false),
    });
};

/**
 * Owns the one simulation event stream. Mounted once from `AppFooter`: every state change lands
 * in the `useSimulation` query cache, and the browser reconnects the stream on its own (the first
 * event after a reconnect is the full current state).
 */
export const useSimulationUpdates = () => {
    const queryClient = useQueryClient();

    useEffect(() => {
        if (typeof EventSource === 'undefined') {
            return;
        }
        let events: EventSource;
        let reopen: ReturnType<typeof setTimeout> | undefined;
        let lastSimulation: string | undefined;
        let reconnected = false;

        const onMessage = (event: MessageEvent<string>) => {
            let simulation: Simulation;
            try {
                simulation = JSON.parse(event.data) as Simulation;
            } catch {
                // A malformed event is ignored; the next one carries the full state again.
                return;
            }
            queryClient.setQueryData(simulationQueryKey(), simulation);
            // Which simulation runs where, not its progress: a replacement can keep the status, and
            // a stream that dropped may have missed changes.
            const current = JSON.stringify([
                simulation.status,
                simulation.project_id,
                simulation.robot_id,
                simulation.started_at,
                simulation.http_url,
            ]);
            // A reconnect refreshes by itself, also when the stream failed before its first event.
            if (reconnected || (lastSimulation !== undefined && current !== lastSimulation)) {
                // Whether a simulated robot is online follows its simulation, and a simulation that
                // starts on a new port stores that address on its robot and moves its IP cameras (and
                // so the environments' cameras) along.
                for (const path of [
                    '/api/projects/{project_id}/robots/online',
                    '/api/projects/{project_id}/robots',
                    '/api/projects/{project_id}/robots/{robot_id}',
                    '/api/projects/{project_id}/cameras',
                    '/api/projects/{project_id}/cameras/{camera_id}',
                    '/api/projects/{project_id}/environments/{environment_id}',
                ]) {
                    queryClient.invalidateQueries({ queryKey: ['get', path] });
                }
            }
            lastSimulation = current;
            reconnected = false;
        };

        const open = () => {
            events = new EventSource(fetchClient.PATH('/api/simulation/events'));
            events.onmessage = onMessage;
            // The browser retries a dropped stream by itself, but gives up for good on an error
            // response (e.g. while the backend restarts): open a new one then.
            events.onerror = () => {
                reconnected = true;
                if (events.readyState === EventSource.CLOSED) {
                    reopen = setTimeout(open, REOPEN_DELAY_MS);
                }
            };
        };
        open();

        return () => {
            clearTimeout(reopen);
            events.close();
        };
    }, [queryClient]);
};

const REOPEN_DELAY_MS = 3_000;

const useSetSimulation = () => {
    const queryClient = useQueryClient();

    return (simulation: Simulation) => queryClient.setQueryData(simulationQueryKey(), simulation);
};

export const useStartSimulation = (project_id: string) => {
    const setSimulation = useSetSimulation();

    return $api.useMutation('post', '/api/simulation', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }]],
        },
        onSuccess: setSimulation,
    });
};

export const useStopSimulation = (project_id?: string) => {
    const setSimulation = useSetSimulation();
    const queryClient = useQueryClient();

    return $api.useMutation('delete', '/api/simulation', {
        meta:
            project_id === undefined
                ? { skipInvalidation: true }
                : {
                      invalidates: [
                          ['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }],
                      ],
                  },
        onSuccess: (simulation) => {
            setSimulation(simulation);
            // The footer's list of the project's simulations.
            void queryClient.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/simulations'] });
        },
    });
};

export const useSimulationLogs = (enabled: boolean) => {
    return $api.useQuery('get', '/api/simulation/logs', undefined, {
        enabled,
        refetchInterval: enabled ? 1_000 : false,
    });
};
