import { Loader, MathUtils } from "three"
import { Canvas, useLoader } from "@react-three/fiber";
import { useEffect } from "react";

import URDFLoader from 'urdf-loader';
import { Grid, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { SchemaEpisode } from "../../api/openapi-spec";

interface RobotProps {
    urdfPath: string
    time: number;
    episode: SchemaEpisode;
}

// Getting typescript of my back. Feel free to fix.
type InputLike = string | string[] | string[][] | Readonly<string | string[] | string[][]>;
type LoaderLike = Loader<any, InputLike>;

function Robot({ urdfPath, time, episode }: RobotProps) {
    const model = useLoader(URDFLoader as unknown as LoaderLike, urdfPath);
    useEffect(() => {
        const frameIndex = Math.floor(time * episode.fps);
        const jointNames = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        if (episode.length > frameIndex) {
            const actionValues = episode.actions[frameIndex].map(MathUtils.degToRad);
            jointNames.forEach((name, index) => {
                model.joints[name].setJointValue(actionValues[index]);
            });
        }
    }, [time, episode])

    useEffect(() => {
        model.rotation.x = -Math.PI / 2
    }, [model]);
    return (
        <primitive object={model} />
    )
}

interface RobotRenderer {
    robot_urdf_path: string;
    time: number;
    episode: SchemaEpisode;
}
export default function RobotRenderer({ robot_urdf_path, time, episode }: RobotRenderer) {
    return (
        <Canvas>
            <PerspectiveCamera makeDefault position={[10, 10, 10]} zoom={10}/>
            <OrbitControls />
            <directionalLight castShadow={true} position={[5, 30, 5]} />
            <ambientLight intensity={Math.PI / 10} />
            <Robot urdfPath={robot_urdf_path} time={time} episode={episode} />
            <Grid infiniteGrid={true} />
        </Canvas>
    )
}
