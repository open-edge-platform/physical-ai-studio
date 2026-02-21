import { createContext, ReactNode, useCallback, useContext, useRef, useState } from 'react';

import { useMutation } from '@tanstack/react-query';
import URDFLoader, { URDFRobot } from 'urdf-loader';

import { SchemaRobotType } from '../../api/openapi-spec';

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
    _setModel: (path: string, model: URDFRobot) => void;
};

const RobotModelsContext = createContext<RobotModelsContextValue>(null);

export const RobotModelsProvider = ({ children }: { children: ReactNode }) => {
    const [models, setModels] = useState<ModelsMap>(() => new Map());

    const getModel = useCallback((path: string) => models.get(path), [models]);
    const hasModel = useCallback((path: string) => models.has(path), [models]);
    const clearModels = useCallback(() => setModels(new Map()), []);

    const _setModel = useCallback(
        (path: string, model: URDFRobot) => {
            setModels((prev) => {
                const next = new Map(prev);
                next.set(path, model);
                return next;
            });
        },
        []
    );

    return (
        <RobotModelsContext.Provider
            value={{
                models,
                getModel,
                hasModel,
                clearModels,
                _setModel,
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
    const { _setModel } = useRobotModels();

    // Track the path being loaded so onSuccess can key it correctly.
    // We use a ref because the mutationFn arg isn't available in onSuccess
    // when using useMutation (variables are on the mutation object, but
    // onSuccess receives (data, variables, context)).
    const pathRef = useRef<string>('');

    return useMutation({
        mutationFn: async (path: string) => {
            pathRef.current = path;

            const loader = new URDFLoader();

            loader.packages = {
                trossen_arm_description: '/widowx',
            };

            return new Promise<URDFRobot>((resolve, reject) => {
                loader.load(path, resolve, console.info, reject);
            });
        },
        onSuccess: async (model) => {
            _setModel(pathRef.current, model);
        },
    });
};
