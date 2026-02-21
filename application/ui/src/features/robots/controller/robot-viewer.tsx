/* eslint-disable react/no-unknown-property */

import { Suspense, useEffect, useRef } from 'react';

import { Grid, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { degToRad } from 'three/src/math/MathUtils.js';
import { URDFRobot } from 'urdf-loader';

import { SchemaRobot, SchemaRobotType } from '../../../api/openapi-spec';
import { useContainerSize } from '../../../components/zoom/use-container-size';
import { urdfPathForType, useLoadModelMutation, useRobotModels } from './../robot-models-context';

// This is a wrapper component for the loaded URDF model
const ActualURDFModel = ({ model }: { model: URDFRobot }) => {
    // Rotate -90 degrees around X-axis (π/2 radians)
    const rotation = [-Math.PI / 2, 0, (-1 * Math.PI) / 4] as const;
    const scale = [3, 3, 3] as const;

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
        if (hasModel(PATH)) return;
        if (loadModelMutation.data || !loadModelMutation.isIdle) return;

        loadModelMutation.mutate(PATH);
    }, [PATH, hasModel, loadModelMutation]);
};

interface RobotViewerProps {
    robot: Pick<SchemaRobot, 'type'>;
    featureValues?: number[];
    featureNames?: string[];
}
export const RobotViewer = ({ robot, featureValues, featureNames }: RobotViewerProps) => {
    const angle = degToRad(-45);

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
                    <ambientLight intensity={0.5} />
                    <directionalLight
                        position={[10, 10, 5]}
                        intensity={1}
                        castShadow
                        shadow-mapSize-width={1024}
                        shadow-mapSize-height={1024}
                    />
                    <PerspectiveCamera makeDefault position={[2.0, 1, 1]} />
                    <OrbitControls />
                    <Grid infiniteGrid cellSize={0.25} sectionColor={'rgb(0, 199, 253)'} fadeDistance={10} />
                    {model && (
                        <group key={model.uuid} position={[0, 0, 0]} rotation={[0, angle, 0]}>
                            <Suspense fallback={null}>
                                <ActualURDFModel model={model} />
                            </Suspense>
                        </group>
                    )}
                </Canvas>
            </div>
        </div>
    );
};
