import { useEffect, useMemo } from 'react';

import { ContactShadows, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { URDFRobot } from 'urdf-loader';

export const SCENE_COLORS = {
    background: '#242528',
    ambientLight: '#c8d6eb',
    primaryLight: '#e8f0ff',
    fillLight: '#88aadd',
    gridCell: '#3a3d4f',
    gridSection: '#545870',
    checkerboardEven: '#2d2f35',
    checkerboardOdd: '#313339',
    trossenReplacement: new THREE.Color('#585858'),
};

const FLOOR_SIZE = 21;
const CHECKERBOARD_TILE_SIZE = 0.5;

export const useConfigureModelShadows = (model: URDFRobot) => {
    useEffect(() => {
        model.traverse((node) => {
            if ((node as THREE.Mesh).isMesh) {
                (node as THREE.Mesh).castShadow = true;
                (node as THREE.Mesh).receiveShadow = true;
            }
        });
    }, [model]);
};

/* eslint-disable react/no-unknown-property */
const CheckerboardFloor = () => {
    const texture = useMemo(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;

        const context = canvas.getContext('2d');
        if (!context) {
            return null;
        }

        const tiles = 24;
        const tileSize = canvas.width / tiles;
        for (let x = 0; x < tiles; x += 1) {
            for (let y = 0; y < tiles; y += 1) {
                context.fillStyle = (x + y) % 2 === 0 ? SCENE_COLORS.checkerboardEven : SCENE_COLORS.checkerboardOdd;
                context.fillRect(x * tileSize, y * tileSize, tileSize, tileSize);
            }
        }

        const checkerboardTexture = new THREE.CanvasTexture(canvas);
        checkerboardTexture.wrapS = THREE.RepeatWrapping;
        checkerboardTexture.wrapT = THREE.RepeatWrapping;
        checkerboardTexture.repeat.set(
            FLOOR_SIZE / (tiles * CHECKERBOARD_TILE_SIZE),
            FLOOR_SIZE / (tiles * CHECKERBOARD_TILE_SIZE)
        );

        return checkerboardTexture;
    }, []);

    useEffect(() => () => texture?.dispose(), [texture]);

    return (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.005, 0]} receiveShadow>
            <planeGeometry args={[FLOOR_SIZE, FLOOR_SIZE]} />
            <meshStandardMaterial map={texture} roughness={0.8} metalness={0} />
        </mesh>
    );
};

export const RobotViewerScene = () => {
    return (
        <>
            <color attach='background' args={[SCENE_COLORS.background]} />
            <ambientLight intensity={0.7} color={SCENE_COLORS.ambientLight} />
            <directionalLight
                position={[1.5, 3.5, 2]}
                intensity={2.5}
                color={SCENE_COLORS.primaryLight}
                castShadow
                shadow-mapSize-width={2048}
                shadow-mapSize-height={2048}
                shadow-camera-left={-6}
                shadow-camera-right={6}
                shadow-camera-top={6}
                shadow-camera-bottom={-6}
                shadow-camera-near={0.2}
                shadow-camera-far={40}
                shadow-bias={-0.0001}
            />
            <directionalLight position={[2, 2, -3]} intensity={0.4} color={SCENE_COLORS.fillLight} />
            <CheckerboardFloor />
            <Grid
                infiniteGrid
                cellSize={0.25}
                cellColor={SCENE_COLORS.gridCell}
                sectionSize={0.5}
                sectionColor={SCENE_COLORS.gridSection}
                fadeDistance={FLOOR_SIZE - 1}
            />
            <ContactShadows position={[0, 0, 0]} opacity={0.2} scale={2.5} blur={2.5} far={1} resolution={512} />
        </>
    );
};
/* eslint-enable react/no-unknown-property */
