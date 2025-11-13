import { v4 as uuidv4 } from 'uuid';

import {
    SchemaProjectConfigOutput,
    SchemaProjectOutput,
    SchemaRobotPortInfo,
    SchemaTeleoperationConfig,
} from '../../../api/openapi-spec';

export const TELEOPERATION_CONFIG_CACHE_KEY = 'teleoperation_config';

export const makeNameSafeForPath = (name: string): string => {
    return name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
};

export const storeConfigToCache = (config: SchemaTeleoperationConfig) => {
    localStorage.setItem(TELEOPERATION_CONFIG_CACHE_KEY, JSON.stringify(config));
};

export const robotConfigMatches = (a: SchemaRobotPortInfo, b: SchemaRobotPortInfo): boolean => {
    return a.serial_id === b.serial_id && a.port == b.port && a.robot_type === b.robot_type;
};

export const configFromCache = (
    cache: SchemaTeleoperationConfig,
    projectConfig: SchemaProjectConfigOutput | undefined | null,
    defaultConfig: SchemaTeleoperationConfig,
    availableRobots: SchemaRobotPortInfo[]
): SchemaTeleoperationConfig => {
    let output = defaultConfig;

    if (projectConfig === undefined || projectConfig === null) {
        return output;
    }

    const problemsInCacheCamera =
        cache.cameras.find((cachedCamera) => {
            const projectCamera = projectConfig.cameras.find((m) => m.name === cachedCamera.name);
            if (projectCamera === undefined) {
                return true;
            }
            const sameProps =
                projectCamera.name === cachedCamera.name &&
                projectCamera.width === cachedCamera.width &&
                projectCamera.height === cachedCamera.height &&
                projectCamera.fps === cachedCamera.fps &&
                projectCamera.use_depth === cachedCamera.use_depth &&
                projectCamera.driver === cachedCamera.driver;

            return !sameProps;
        }) !== undefined;

    if (!problemsInCacheCamera) {
        output = { ...output, cameras: cache.cameras };
    }

    const problemsInFollower =
        cache.leader.robot_type !== projectConfig.robot_type &&
        !!availableRobots.find((b) => robotConfigMatches(cache.leader, b));

    if (!problemsInFollower) {
        output = { ...output, follower: cache.follower };
    }

    const problemsInLeader =
        cache.leader.robot_type !== projectConfig.robot_type &&
        !!availableRobots.find((b) => robotConfigMatches(cache.leader, b));

    if (!problemsInLeader) {
        output = { ...output, leader: cache.leader };
    }

    return output;
};

export const initialTeleoperationConfig = (
    initialTask: string,
    project: SchemaProjectOutput,
    dataset_id: string | undefined,
    availableRobots: SchemaRobotPortInfo[]
): SchemaTeleoperationConfig => {
    const cachedConfig = localStorage.getItem(TELEOPERATION_CONFIG_CACHE_KEY);
    const config: SchemaTeleoperationConfig = {
        task: initialTask,
        fps: project.config?.fps ?? 30,
        dataset: project.datasets.find((d) => d.id === dataset_id) ?? {
            project_id: project.id!,
            name: '',
            path: '',
            id: uuidv4(),
        },
        cameras: project.config?.cameras ?? [],
        follower: {
            id: '',
            robot_type: project.config?.robot_type ?? '',
            serial_id: '',
            port: '',
            type: 'follower',
        },
        leader: {
            id: '',
            robot_type: project.config?.robot_type ?? '',
            serial_id: '',
            port: '',
            type: 'leader',
        },
    };
    if (cachedConfig !== null) {
        const cache = JSON.parse(cachedConfig) as SchemaTeleoperationConfig;
        return configFromCache(cache, project.config, config, availableRobots);
    }
    return config;
};
