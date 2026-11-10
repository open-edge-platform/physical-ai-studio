/* eslint-disable react/no-unknown-property */

import { Suspense, useMemo } from 'react';

import { DimensionValue, Flex, Text, View } from '@geti-ui/ui';
import { Bounds, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { degToRad } from 'three/src/math/MathUtils.js';
import { URDFRobot } from 'urdf-loader';

import { useLoadModelQuery } from '../../robot-models-context';
import { SCENE_COLORS } from '../../robot-viewer-scene';

const ThumbnailModel = ({ model }: { model: URDFRobot }) => {
    // urdf-loader models are cached and shared; clone so the same model can sit in several canvases.
    const clone = useMemo(() => {
        const copy = model.clone(true);
        // Same trick as RobotViewer: Trossen's black texture is unreadable on the dark background.
        copy.traverse((node) => {
            const mesh = node as THREE.Mesh;
            if (!mesh.isMesh || Array.isArray(mesh.material)) return;
            if (mesh.material.name.toLowerCase().includes('trossen_black')) {
                const material = mesh.material.clone() as THREE.MeshPhongMaterial;
                material.map = null;
                material.color.copy(SCENE_COLORS.trossenReplacement);
                mesh.material = material;
            }
        });
        return copy;
    }, [model]);

    return (
        <group rotation={[0, degToRad(-45), 0]}>
            <group rotation={[-Math.PI / 2, 0, -Math.PI / 4]} scale={[3, 3, 3]}>
                <primitive object={clone} />
            </group>
        </group>
    );
};

const Overlay = ({ children }: { children: string }) => (
    <View position='absolute' top={0} bottom={0} left={0} right={0}>
        <Flex height='100%' alignItems='center' justifyContent='center'>
            <Text>{children}</Text>
        </Flex>
    </View>
);

const ThumbnailCanvas = ({ type, isActive }: { type: string; isActive: boolean }) => {
    const { data: model, isPending, error } = useLoadModelQuery(type);

    return (
        <>
            <Canvas frameloop={isActive ? 'always' : 'demand'} dpr={[1, 1.5]}>
                <color attach='background' args={[SCENE_COLORS.background]} />
                <ambientLight intensity={0.9} color={SCENE_COLORS.ambientLight} />
                <directionalLight position={[1.5, 3.5, 2]} intensity={2.2} color={SCENE_COLORS.primaryLight} />
                <directionalLight position={[2, 2, -3]} intensity={0.5} color={SCENE_COLORS.fillLight} />
                <PerspectiveCamera makeDefault position={[1.6, 0.9, 0.9]} />
                <OrbitControls
                    makeDefault
                    enableZoom={false}
                    enablePan={false}
                    enableRotate={false}
                    autoRotate={isActive}
                    autoRotateSpeed={4}
                />
                {model && (
                    <Bounds fit clip observe margin={1.15}>
                        <ThumbnailModel model={model} />
                    </Bounds>
                )}
            </Canvas>
            {isPending && <Overlay>Loading…</Overlay>}
            {error && <Overlay>No preview</Overlay>}
        </>
    );
};

/** Small, non-interactive URDF preview for cards. Spins while `isActive`. */
export const RobotThumbnail = ({
    type,
    isActive = false,
    height = 'size-2400',
}: {
    type: string;
    isActive?: boolean;
    height?: DimensionValue;
}) => (
    <View position='relative' width='100%' height={height} overflow='hidden' borderRadius='regular'>
        <Suspense fallback={<Overlay>Loading…</Overlay>}>
            <ThumbnailCanvas type={type} isActive={isActive} />
        </Suspense>
    </View>
);
