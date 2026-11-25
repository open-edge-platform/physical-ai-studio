// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti-ui/ui';

import { $api } from '../../../../api/client';
import { SchemaSimulationCameraInput } from '../../../../api/openapi-spec';
import { useProjectId } from '../../../projects/use-project';
import { NameField } from '../components/name-field';
import { useCameraFormFields } from '../components/use-camera-form-fields';
import { DriverFormSchema } from '../provider';

/** Only added from a running simulation, never configured here: the form just renames one. */
export const initialSimulationState: DriverFormSchema<'simulation'> = {
    driver: 'simulation',
    hardware_name: null,
};

export const validateSimulation = (
    formData: DriverFormSchema<'simulation'>
): formData is SchemaSimulationCameraInput => {
    return !!formData.name && !!formData.fingerprint && !!formData.payload?.robot_id && !!formData.payload?.camera;
};

export const SimulationFormFields = () => {
    const { project_id } = useProjectId();
    const { formData, updateField } = useCameraFormFields('simulation');
    const robot_id = formData.payload?.robot_id ?? '';
    const { data: robot } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}',
        { params: { path: { project_id, robot_id } } },
        { enabled: robot_id !== '' }
    );

    return (
        <Flex gap='size-100' direction='column'>
            <NameField value={formData.name ?? ''} onChange={(name) => updateField('name', name)} />
            <Text>
                Camera “{formData.payload?.camera}” of the simulation of {robot?.name ?? 'its robot'}. It streams while
                that simulation runs.
            </Text>
        </Flex>
    );
};
