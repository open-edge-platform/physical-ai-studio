import { Suspense, useEffect, useRef } from 'react';

import { Grid, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
//import { useControls } from 'leva';
import { URDFRobot } from 'urdf-loader';

import { useContainerSize } from '../../../components/zoom/use-container-size';
import { useAction, useLoadModelMutation } from './../action-context';

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

const useLoadSO101 = () => {
    const loadModelMutation = useLoadModelMutation();
    const { models } = useAction();

    const PATH = '/SO101/so101_new_calib.urdf';

    const ref = useRef(false);
    useEffect(() => {
        if (models.length > 0) {
            return;
        }

        if (loadModelMutation.data || !loadModelMutation.isIdle) {
            return;
        }

        if (ref.current) {
            return;
        }

        ref.current = true;
        loadModelMutation.mutate(PATH);
    }, [models]);
};

export const RobotViewer = () => {
    const angle = 10;
    useLoadSO101();
    const ref = useRef<HTMLDivElement>(null);
    const size = useContainerSize(ref);
    const { models } = useAction();
    const model = models.at(0);

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
                    <Grid infiniteGrid cellSize={0.25} sectionColor={'rgb(0, 199, 253)'} />
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
