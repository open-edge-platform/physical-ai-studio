import { SchemaRobotCatalogDefinitionResponse } from '../../api/openapi-spec';

export const getMockedRobotCatalogDefinition = (
    overrides: Partial<SchemaRobotCatalogDefinitionResponse> = {}
): SchemaRobotCatalogDefinitionResponse => {
    const type = overrides.type ?? 'SO101_Follower';

    return {
        type,
        display_name: type.replaceAll('_', ' '),
        role: 'follower',
        urdf_path: `/api/robots/catalog/${type}/urdf`,
        package_map: {},
        joint_map: {},
        zero_calibration: null,
        ...overrides,
    };
};
