/* eslint-disable react/no-unknown-property */

import { Suspense, useEffect, useRef } from 'react';

import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { degToRad } from 'three/src/math/MathUtils.js';
import { URDFRobot } from 'urdf-loader';

import { useContainerSize } from '../../../components/zoom/use-container-size';
import { useRobotCatalogDefinitionQuery } from '../robot-catalog.hooks';
import { SchemaRobot } from '../robot-types';
import { mapJointToURDFJoint, useLoadModelQuery } from './../robot-models-context';
import { RobotViewerScene, SCENE_COLORS, useConfigureModelShadows } from './../robot-viewer-scene';

import classes from './robot-viewer.module.css';

/** Material name used by the dark parts in the Trossen URDF. */
const TROSSEN_DARK_MATERIAL = 'trossen_black';

/**
 * Find the shared `trossen_black` material on the model and replace its dark
 * texture with a solid color.
 *
 * The model is guaranteed to have all its STL meshes loaded before it enters
 * React state (see `useLoadModelQuery` which resolves on
 * `LoadingManager.onLoad`), so a plain `useEffect` is sufficient here.
 *
 * Because urdf-loader uses a shared material instance for each named material,
 * mutating it in-place ensures all meshes (even nested deep in the tree) pick
 * up the change.  Originals are restored on cleanup.
 */
const useBrightenDarkMaterials = (model: URDFRobot | undefined, enabled: boolean) => {
    useEffect(() => {
        if (!model || !enabled) return;

        const saved: {
            mat: THREE.MeshPhongMaterial;
            map: THREE.Texture | null;
            color: THREE.Color;
        }[] = [];

        const seen = new Set<THREE.Material>();

        model.traverse((node) => {
            if (!(node as THREE.Mesh).isMesh) {
                return;
            }
            const mesh = node as THREE.Mesh;
            const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];

            for (const mat of materials) {
                if (seen.has(mat)) {
                    continue;
                }

                seen.add(mat);

                if (!mat.name.toLowerCase().includes(TROSSEN_DARK_MATERIAL)) {
                    continue;
                }

                const phong = mat as THREE.MeshPhongMaterial;
                saved.push({ mat: phong, map: phong.map, color: phong.color.clone() });

                phong.map = null;
                phong.color.copy(SCENE_COLORS.trossenReplacement);
                phong.needsUpdate = true;
            }
        });

        return () => {
            for (const s of saved) {
                s.mat.map = s.map;
                s.mat.color.copy(s.color);
                s.mat.needsUpdate = true;
            }
        };
    }, [model, enabled]);
};

// This is a wrapper component for the loaded URDF model
const ActualURDFModel = ({ model, isTrossen }: { model: URDFRobot; isTrossen: boolean }) => {
    // Rotate -90 degrees around X-axis (π/2 radians)
    const rotation = [-Math.PI / 2, 0, (-1 * Math.PI) / 4] as const;
    const scale = [3, 3, 3] as const;

    useBrightenDarkMaterials(model, isTrossen);
    useConfigureModelShadows(model);

    return (
        <group rotation={rotation} scale={scale}>
            <primitive object={model} />
        </group>
    );
};

interface RobotViewerProps {
    robot: Pick<SchemaRobot, 'type'>;
    featureValues?: number[];
    featureNames?: string[];
}
export const RobotViewer = ({ robot = { type: 'SO101_Follower' }, featureValues, featureNames }: RobotViewerProps) => {
    const angle = degToRad(-45);
    const isTrossen = robot.type.toLowerCase().includes('trossen');

    const { data: definition } = useRobotCatalogDefinitionQuery(robot.type);
    const jointMap = definition.joint_map;

    const { data: model, error, isPending } = useLoadModelQuery(robot.type);
    const ref = useRef<HTMLDivElement>(null);
    const size = useContainerSize(ref);

    useEffect(() => {
        if (featureValues !== undefined && featureNames !== undefined && model !== undefined) {
            featureNames.forEach((_, index) => {
                mapJointToURDFJoint(
                    {
                        name: featureNames[index],
                        value: featureValues[index],
                    },
                    model,
                    jointMap
                );
            });
        }
    }, [featureValues, featureNames, model, jointMap]);

    return (
        <div ref={ref} className={classes.viewer}>
            <div className={classes.canvas} style={{ height: `${size.height}px`, width: `${size.width}px` }}>
                <Canvas shadows>
                    <RobotViewerScene />
                    <PerspectiveCamera makeDefault position={[2.0, 1, 1]} />
                    <OrbitControls enableDamping={false} />
                    {model && (
                        <group key={model.uuid} position={[0, 0, 0]} rotation={[0, angle, 0]}>
                            <Suspense fallback={null}>
                                <ActualURDFModel model={model} isTrossen={isTrossen} />
                            </Suspense>
                        </group>
                    )}
                </Canvas>
                {isPending && <div className={classes.loadingOverlay}>Loading robot model...</div>}
                {error && (
                    <div className={classes.errorOverlay} role='alert'>
                        Failed to load robot model: {error instanceof Error ? error.message : String(error)}
                    </div>
                )}
            </div>
        </div>
    );
};
