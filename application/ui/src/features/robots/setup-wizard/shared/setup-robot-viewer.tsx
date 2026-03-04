/* eslint-disable react/no-unknown-property */

import { Suspense, useEffect, useMemo, useRef } from 'react';

import { Grid, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { degToRad } from 'three/src/math/MathUtils.js';
import { URDFRobot } from 'urdf-loader';

import { SchemaRobotType } from '../../../../api/openapi-spec';
import { useContainerSize } from '../../../../components/zoom/use-container-size';
import { urdfPathForType, useLoadModelMutation, useRobotModels } from '../../robot-models-context';
import { JointHighlight, useJointHighlight } from './use-joint-highlight';

// ---------------------------------------------------------------------------
// Inner model component — renders the URDF and applies joint highlight
// ---------------------------------------------------------------------------

const ActualURDFModel = ({ model, highlights }: { model: URDFRobot; highlights: JointHighlight[] }) => {
    const rotation = [-Math.PI / 2, 0, (-1 * Math.PI) / 4] as const;
    const scale = [3, 3, 3] as const;

    useJointHighlight(model, highlights);

    return (
        <group rotation={rotation} scale={scale}>
            <primitive object={model} />
        </group>
    );
};

// ---------------------------------------------------------------------------
// Camera controller — smoothly focuses on highlighted joint
// ---------------------------------------------------------------------------

/** Exponential damping speed — higher = snappier */
const LERP_SPEED = 3;

/**
 * Threshold below which we stop lerping (squared distance).
 * Prevents perpetual micro-adjustments.
 */
const EPSILON_SQ = 0.00001;

interface CameraControllerProps {
    controlsRef: React.RefObject<OrbitControlsImpl | null>;
    model: URDFRobot | undefined;
    highlights: JointHighlight[];
}

/**
 * Smoothly animates the camera and OrbitControls target to focus on the
 * highlighted joint. When multiple joints are highlighted, the camera is
 * left untouched. When no joint is highlighted, the camera is also left
 * untouched (user can freely orbit).
 *
 * Must be rendered inside `<Canvas>` (uses `useFrame` and `useThree`).
 */
const CameraController = ({ controlsRef, model, highlights }: CameraControllerProps) => {
    const { camera } = useThree();

    // Compute desired camera position + look-at target for a single highlighted joint.
    // Returns null when there's nothing to focus on (empty or multiple) — the camera stays put.
    const focus = useMemo(() => {
        // Only focus camera for a single joint — for multiple joints leave camera alone
        if (!model || highlights.length !== 1) {
            return null;
        }

        const joint = model.joints[highlights[0].joint];
        if (!joint) {
            return null;
        }

        // Get the joint's world-space position (accounts for all parent transforms)
        const worldPos = new THREE.Vector3();
        joint.getWorldPosition(worldPos);

        // Compute a bounding sphere of the joint's child link for framing distance
        const childLink = joint.children.find((c: THREE.Object3D & { isURDFLink?: boolean }) => c.isURDFLink);
        let radius = 0.3; // fallback
        if (childLink) {
            const box = new THREE.Box3().setFromObject(childLink);
            const sphere = new THREE.Sphere();
            box.getBoundingSphere(sphere);
            radius = Math.max(sphere.radius, 0.15);
        }

        // Offset camera: slightly above and to the side of the joint
        const distance = radius * 4 + 0.3;
        const offset = new THREE.Vector3(distance * 0.7, distance * 0.5, distance * 0.7);

        return {
            position: worldPos.clone().add(offset),
            target: worldPos.clone(),
        };
    }, [model, highlights]);

    useFrame((_, delta) => {
        // Only animate the camera when we have a joint to focus on
        if (!focus || !controlsRef.current) {
            return;
        }

        const t = 1 - Math.exp(-LERP_SPEED * delta);

        // Only lerp if we're not already at the target (avoid micro-jitter)
        const posDist = camera.position.distanceToSquared(focus.position);
        const tgtDist = controlsRef.current.target.distanceToSquared(focus.target);

        if (posDist > EPSILON_SQ || tgtDist > EPSILON_SQ) {
            camera.position.lerp(focus.position, t);
            controlsRef.current.target.lerp(focus.target, t);
            controlsRef.current.update();
        }
    });

    return null;
};

// ---------------------------------------------------------------------------
// URDF loading hook (same as robot-viewer.tsx)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Public component
// ---------------------------------------------------------------------------

interface SetupRobotViewerProps {
    robotType: SchemaRobotType;
    /** Joints to highlight with per-joint colors. Pass an empty array for no highlight. */
    highlights?: JointHighlight[];
}

/**
 * 3D robot viewer for the setup wizard. Based on the shared RobotViewer but
 * adds support for highlighting individual joints via emissive material glow.
 *
 * Animations (centering, range-of-motion) and live position syncing are
 * handled externally via the shared `useRobotModels()` context — this
 * component just renders the model and applies the highlight.
 */
export const SetupRobotViewer = ({ robotType, highlights = [] }: SetupRobotViewerProps) => {
    const angle = degToRad(-45);

    const PATH = urdfPathForType(robotType);
    useLoadURDF(robotType);
    const ref = useRef<HTMLDivElement>(null);
    const controlsRef = useRef<OrbitControlsImpl>(null);
    const size = useContainerSize(ref);
    const { getModel } = useRobotModels();
    const model = getModel(PATH);

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
                    <OrbitControls ref={controlsRef} />
                    <CameraController controlsRef={controlsRef} model={model} highlights={highlights} />
                    <Grid infiniteGrid cellSize={0.25} sectionColor={'rgb(0, 199, 253)'} fadeDistance={10} />
                    {model && (
                        <group key={model.uuid} position={[0, 0, 0]} rotation={[0, angle, 0]}>
                            <Suspense fallback={null}>
                                <ActualURDFModel model={model} highlights={highlights} />
                            </Suspense>
                        </group>
                    )}
                </Canvas>
            </div>
        </div>
    );
};
