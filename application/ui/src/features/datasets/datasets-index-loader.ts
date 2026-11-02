// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { redirect, type LoaderFunctionArgs } from 'react-router';

import { fetchClient } from '../../api/client';
import { paths } from '../../paths';
import { clearLastUsedDatasetId, getLastUsedDatasetId, setLastUsedDatasetId } from './last-used-dataset';

export const recordLastUsedDatasetLoader = ({ params }: LoaderFunctionArgs): null => {
    const { project_id, dataset_id } = params;

    if (project_id !== undefined && dataset_id !== undefined) {
        setLastUsedDatasetId(project_id, dataset_id);
    }

    return null;
};

export const datasetsIndexLoader = async ({ params }: LoaderFunctionArgs) => {
    const project_id = params.project_id;

    if (project_id === undefined) {
        return redirect(paths.projects.index({}));
    }

    try {
        const { data: projectData, error } = await fetchClient.GET('/api/projects/{project_id}', {
            params: { path: { project_id } },
        });

        if (error !== undefined || projectData === undefined) {
            return null;
        }

        const lastUsedDatasetId = getLastUsedDatasetId(project_id);
        const dataset = projectData.datasets.find(({ id }) => id === lastUsedDatasetId);

        if (dataset?.id !== undefined) {
            return redirect(paths.project.datasets.show({ project_id, dataset_id: dataset.id }));
        }

        if (lastUsedDatasetId !== undefined) {
            clearLastUsedDatasetId(project_id);
        }

        const firstDatasetId = projectData.datasets.at(0)?.id;
        if (firstDatasetId === undefined) {
            return null;
        }

        return redirect(
            paths.project.datasets.show({
                project_id,
                dataset_id: firstDatasetId,
            })
        );
    } catch {
        return null;
    }
};
