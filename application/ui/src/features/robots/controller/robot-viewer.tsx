/* eslint-disable react/no-unknown-property */

import { Suspense, useEffect, useRef } from 'react';

import { Grid, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { degToRad } from 'three/src/math/MathUtils.js';
import { URDFRobot } from 'urdf-loader';

import { SchemaRobot, SchemaRobotType } from '../../../api/openapi-spec';
import { useContainerSize } from '../../../components/zoom/use-container-size';
import { urdfPathForType, useLoadModelMutation, useRobotModels } from './../robot-models-context';

/** Material name used by the dark parts in the Trossen URDF. */
const TROSSEN_DARK_MATERIAL = 'trossen_black';

/** Replacement color for dark Trossen materials. */
const TROSSEN_REPLACEMENT_COLOR = new THREE.Color('#585858');

/**
 * Find the shared `trossen_black` material on the model and replace its dark
 * texture with a solid color.
 *
 * The model is guaranteed to have all its STL meshes loaded before it enters
 * React state (see `useLoadModelMutation` which resolves on
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
                phong.color.copy(TROSSEN_REPLACEMENT_COLOR);
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

    return (
        <group rotation={rotation} scale={scale}>
            <primitive object={model} />
        </group>
    );
};

const useLoadURDF = (robotType: SchemaRobotType) => {
    const loadModelMutation = useLoadModelMutation();
    const { hasModel } = useRobotModels();

    const PATH = urdfPathForType(robotType);

    useEffect(() => {
        if (hasModel(PATH)) {
            return;
        }

        loadModelMutation.mutate(PATH);
    }, [PATH, hasModel, loadModelMutation]);
};

interface RobotViewerProps {
    robot: Pick<SchemaRobot, 'type'>;
    featureValues?: number[];
    featureNames?: string[];
}
export const RobotViewer = ({ robot = { type: 'SO101_Follower' }, featureValues, featureNames }: RobotViewerProps) => {
    const angle = degToRad(-45);
    const isTrossen = robot.type.toLowerCase().includes('trossen');

    const PATH = urdfPathForType(robot.type);
    useLoadURDF(robot.type);
    const ref = useRef<HTMLDivElement>(null);
    const size = useContainerSize(ref);
    const { getModel } = useRobotModels();
    const model = getModel(PATH);

    useEffect(() => {
        if (featureValues !== undefined && featureNames !== undefined && model !== undefined) {
            featureNames.forEach((name, index) => {
                if (index < featureValues.length && name.endsWith('.pos')) {
                    const joint_name = name.replace('.pos', '');

                    if (joint_name === 'gripper' && model.robotName == 'wxai') {
                        model.setJointValue('left_carriage_joint', featureValues[index]); // meters
                    } else if (model.joints[joint_name] != undefined) {
                        model.joints[joint_name].setJointValue(degToRad(featureValues[index]));
                    }
                }
            });
        }
    }, [featureValues, featureNames, model]);

    return (
        <div ref={ref} style={{ width: '100%', height: '100%' }}>
            <div className='canvas-container' style={{ height: `${size.height}px`, width: `${size.width}px` }}>
                <Canvas shadows>
                    <color attach='background' args={['#242528']} />
                    <ambientLight intensity={0.4} />
                    <directionalLight
                        position={[10, 10, 5]}
                        intensity={1}
                        castShadow
                        shadow-mapSize-width={1024}
                        shadow-mapSize-height={1024}
                    />
                    <directionalLight position={[-5, 5, -5]} intensity={0.3} />
                    <directionalLight position={[0, -3, 5]} intensity={0.2} />
                    <PerspectiveCamera makeDefault position={[2.0, 1, 1]} />
                    <OrbitControls />
                    <Grid infiniteGrid cellSize={0.25} sectionColor={'rgb(0, 199, 253)'} fadeDistance={10} />
                    {model && (
                        <group key={model.uuid} position={[0, 0, 0]} rotation={[0, angle, 0]}>
                            <Suspense fallback={null}>
                                <ActualURDFModel model={model} isTrossen={isTrossen} />
                            </Suspense>
                        </group>
                    )}
                </Canvas>
            </div>
        </div>
    );
};
