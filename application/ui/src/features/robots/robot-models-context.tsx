import { createContext, ReactNode, useCallback, useContext, useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';
import * as THREE from 'three';
import { degToRad } from 'three/src/math/MathUtils.js';
import URDFLoader, { URDFRobot } from 'urdf-loader';

import { SchemaRobotType } from './robot-types';

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/** Resolve a `SchemaRobotType` to its URDF asset path. */
export const urdfPathForType = (robotType: SchemaRobotType): string => {
    if (robotType !== undefined && robotType.toLowerCase().includes('trossen')) {
        return '/widowx/urdf/generated/wxai/wxai_follower.urdf';
    }
    return '/SO101/so101_new_calib.urdf';
};

const SO101_TO_URDF = {
    'shoulder_pan.pos': ['shoulder_pan'],
    'shoulder_lift.pos': ['shoulder_lift'],
    'elbow_flex.pos': ['elbow_flex'],
    'wrist_flex.pos': ['wrist_flex'],
    'wrist_roll.pos': ['wrist_roll'],
    'gripper.pos': ['gripper'],
};
const TROSSEN_TO_URDF = {
    'shoulder_pan.pos': ['joint_0'],
    'shoulder_lift.pos': ['joint_1'],
    'elbow_flex.pos': ['joint_2'],
    'wrist_flex.pos': ['joint_3'],
    'wrist_yaw.pos': ['joint_4'],
    'wrist_roll.pos': ['joint_5'],
    'gripper.pos': ['left_carriage_joint', 'right_carriage_joint'],
};

const ROBOT_TYPE_TO_URDF_MAP: Record<SchemaRobotType, Record<string, string[]>> = {
    SO101_Follower: SO101_TO_URDF,
    SO101_Leader: SO101_TO_URDF,
    Trossen_WidowXAI_Leader: TROSSEN_TO_URDF,
    Trossen_WidowXAI_Follower: TROSSEN_TO_URDF,
};

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
    /** @internal — used only by `useLoadModelMutation`. */
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

export const useLoadModelMutation = () => {
    const { setModel } = useRobotModels();

    // Track the path being loaded so onSuccess can key it correctly.
    // We use a ref because the mutationFn arg isn't available in onSuccess
    // when using useMutation (variables are on the mutation object, but
    // onSuccess receives (data, variables, context)).
    const pathRef = useRef<string>('');

    return useMutation({
        mutationFn: async (path: string) => {
            pathRef.current = path;

            // Use a custom LoadingManager so the promise only resolves after
            // all STL meshes have finished loading — not just after the URDF
            // XML is parsed.  URDFLoader.load() calls onComplete as soon as
            // parse() returns, but STL files are fetched asynchronously via
            // the manager.  By resolving on manager.onLoad we guarantee the
            // model's mesh children exist in the scene graph.
            const manager = new THREE.LoadingManager();
            const loader = new URDFLoader(manager);

            loader.packages = {
                trossen_arm_description: '/widowx',
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
        },
        onSuccess: async (model) => {
            setModel(pathRef.current, model);
        },
        meta: { skipInvalidation: true },
    });
};
