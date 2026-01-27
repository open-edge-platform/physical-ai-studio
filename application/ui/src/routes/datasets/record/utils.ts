import {
    SchemaDatasetOutput,
    SchemaEnvironmentWithRelations,
    SchemaProjectOutput,
    SchemaTeleoperationConfig,
} from '../../../api/openapi-spec';

export const TELEOPERATION_CONFIG_CACHE_KEY = 'teleoperation_config';

export const makeNameSafeForPath = (name: string): string => {
    return name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
};

export const initialTeleoperationConfig = (
    initialTask: string,
    dataset: SchemaDatasetOutput,
    availableEnvironments: SchemaEnvironmentWithRelations[]
): SchemaTeleoperationConfig => {
    return {
        task: initialTask,
        dataset,
        environment: availableEnvironments[0]
    };
};
