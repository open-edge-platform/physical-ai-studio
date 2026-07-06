import { createContext, ReactNode, useCallback, useContext, useEffect, useState } from 'react';

import { useQuery } from '@tanstack/react-query';
import * as THREE from 'three';
import { degToRad } from 'three/src/math/MathUtils.js';
import URDFLoader, { URDFRobot } from 'urdf-loader';

import { SchemaRobotType } from './robot-types';
import { ROBOT_TYPE_TO_URDF_MAP } from './robots-configuration';

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

export const mapJointToURDFJoint = (
    joint: { name: string; value: number },
    model: URDFRobot,
    robotType: SchemaRobotType
) => {
    if (!joint.name.endsWith('.pos')) {
        return;
    }
    const modelJointMap = ROBOT_TYPE_TO_URDF_MAP[robotType];
    const modelJoints = modelJointMap[joint.name] ?? [];

    modelJoints.forEach((modelJointName) => {
        const isRevolute = model.joints[modelJointName].jointType === 'revolute';

        model.setJointValue(modelJointName, isRevolute ? degToRad(joint.value) : joint.value); // meters
    });
};

/**
 * Models are stored in a Map keyed by the URDF path that was loaded.
 * This gives O(1) lookup, prevents duplicates, and caches previously-loaded
 * models so switching between robot types doesn't re-fetch.
 */
type ModelsMap = Map<string, URDFRobot>;

type RobotModelsContextValue = null | {
    /** Get a cached model by its URDF path. */
    getModel: (path: string) => URDFRobot | undefined;
    /** Check whether a model has been loaded for a given URDF path. */
    hasModel: (path: string) => boolean;
    /** Remove all cached models. */
    clearModels: () => void;
    /**
     * The underlying Map — exposed for the rare case where a consumer needs
     * to iterate (e.g. animations). Prefer `getModel` / `hasModel` instead.
     */
    models: ModelsMap;
    /** @internal — used only by `useLoadModelQuery`. */
    setModel: (path: string, model: URDFRobot) => void;
};

const RobotModelsContext = createContext<RobotModelsContextValue>(null);

export const RobotModelsProvider = ({ children }: { children: ReactNode }) => {
    const [models, setModels] = useState<ModelsMap>(() => new Map());

    const getModel = useCallback((path: string) => models.get(path), [models]);
    const hasModel = useCallback((path: string) => models.has(path), [models]);
    const clearModels = useCallback(() => setModels(new Map()), []);

    const setModel = useCallback((path: string, model: URDFRobot) => {
        setModels((prev) => {
            const next = new Map(prev);
            next.set(path, model);
            return next;
        });
    }, []);

    return (
        <RobotModelsContext.Provider
            value={{
                models,
                getModel,
                hasModel,
                clearModels,
                setModel,
            }}
        >
            {children}
        </RobotModelsContext.Provider>
    );
};

export const useRobotModels = () => {
    return useContext(RobotModelsContext)!;
};

export const useLoadModelQuery = (path: string) => {
    const { getModel, setModel } = useRobotModels();

    const cachedModel = getModel(path);
    const query = useQuery({
        queryKey: ['robotModel', path],
        queryFn: () => cachedModel ?? loadURDFModel(path),
        initialData: cachedModel,
        staleTime: Infinity,
        gcTime: 1000 * 60 * 30,
        enabled: path !== '' && !cachedModel,
    });

    useEffect(() => {
        if (query.data && getModel(path) !== query.data) {
            setModel(path, query.data);
        }
    }, [getModel, path, query.data, setModel]);

    return query;
};

const loadURDFModel = async (path: string): Promise<URDFRobot> => {
    if (!path) {
        throw new Error('Path is required');
    }

    // Use a custom LoadingManager so the promise only resolves after all STL
    // meshes have finished loading, not just after the URDF XML is parsed.
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);

    loader.packages = {
        trossen_arm_description: '/api/robots/catalog/Trossen_WidowXAI_Leader',
    };

    return new Promise<URDFRobot>((resolve, reject) => {
        let model: URDFRobot | null = null;

        manager.onLoad = () => {
            if (model) {
                resolve(model);
            }
        };
        manager.onError = (url) => {
            reject(new Error(`Failed to load: ${url}`));
        };

        loader.load(
            path,
            (result) => {
                model = result;
            },
            undefined,
            reject
        );
    });
};
