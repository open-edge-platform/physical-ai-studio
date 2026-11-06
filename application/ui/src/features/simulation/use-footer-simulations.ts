// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../api/client';
import { useOptionalProjectId } from '../projects/use-project';
import { useRobotCatalogQuery } from '../robots/robot-catalog.hooks';
import { launchNoun } from './robot-simulation';
import { isActiveSimulation, useSimulation } from './use-simulation';

export interface FooterSimulation {
    key: string;
    label: string;
    scene: string | null;
    startedByStudio: boolean;
    viewerUrl: string | null;
    /** What it is called (`launchNoun`): "simulation", "VR leader"... */
    noun: string;
}

/**
 * The simulations to list in the footer: Studio's own (from any project) and those that run for
 * the current project's simulated robots, also when started elsewhere (CLI).
 */
export const useFooterSimulations = (): FooterSimulation[] => {
    const { project_id } = useOptionalProjectId();
    const { data: studio } = useSimulation();
    const { data: catalog } = useRobotCatalogQuery();
    const nounOf = (robotType: string | null | undefined) =>
        launchNoun(catalog.find(({ type }) => type === robotType)?.simulation);
    const { data: project } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/simulations',
        { params: { path: { project_id: project_id ?? '' } } },
        { enabled: project_id !== undefined, refetchInterval: 5_000 }
    );

    const rows: FooterSimulation[] = (project_id === undefined ? [] : (project ?? [])).map((simulation) => ({
        key: simulation.robot_id,
        label: simulation.robot_name,
        scene: simulation.scene ?? null,
        startedByStudio: simulation.started_by === 'studio',
        viewerUrl: simulation.viewer_proxy_url ?? null,
        noun: nounOf(simulation.robot_type),
    }));

    if (isActiveSimulation(studio) && !rows.some(({ key }) => key === studio.robot_id)) {
        rows.unshift({
            key: studio.robot_id ?? 'studio',
            label: (studio.robot_type ?? 'Simulation').replaceAll('_', ' '),
            scene: studio.scene ?? null,
            startedByStudio: true,
            viewerUrl: studio.viewer_proxy_url ?? null,
            noun: nounOf(studio.robot_type),
        });
    }

    return rows;
};
