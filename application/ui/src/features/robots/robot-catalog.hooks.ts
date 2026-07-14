import { $api } from '../../api/client';
import { SchemaRobotCatalogDefinitionResponse, SchemaRobotType } from '../../api/openapi-spec';
import { SchemaRobot } from './robot-types';

export const useRobotCatalogQuery = () => {
    return $api.useSuspenseQuery('get', '/api/robots/catalog', {
        meta: { skipInvalidation: true },
    });
};

export const useRobotCatalogDefinitionQuery = (robotType: SchemaRobotType) => {
    return $api.useSuspenseQuery(
        'get',
        '/api/robots/catalog',
        {
            meta: { skipInvalidation: true },
        },
        {
            select: (data): SchemaRobotCatalogDefinitionResponse => {
                const definition = data.find(({ type }) => type === robotType);

                if (definition === undefined) {
                    throw new Error(`Missing catalog entry for robot type: ${robotType}`);
                }
                return definition;
            },
        }
    );
};

const useRobotCatalogMap = () => {
    const query = useRobotCatalogQuery();

    const byType = new Map<SchemaRobotType, SchemaRobotCatalogDefinitionResponse>();
    query.data.forEach((entry) => {
        byType.set(entry.type, entry);
    });

    return byType;
};

export const useIsRobotRole = () => {
    const byType = useRobotCatalogMap();

    const isFollower = (robot: Pick<SchemaRobot, 'type'>) => {
        const entry = byType.get(robot.type);
        if (entry === undefined) {
            throw new Error(`Missing catalog entry for robot type: ${robot.type}`);
        }
        return entry.role === 'follower';
    };

    const isLeader = (robot: Pick<SchemaRobot, 'type'>) => {
        const entry = byType.get(robot.type);
        if (entry === undefined) {
            throw new Error(`Missing catalog entry for robot type: ${robot.type}`);
        }
        return entry.role === 'leader';
    };

    return {
        isFollower,
        isLeader,
    };
};
