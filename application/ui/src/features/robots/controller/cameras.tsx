import { CameraThumbnail } from '../camera-thumbnail';
import { useRobot } from '../use-robot';

export const Cameras = () => {
    const robot = useRobot();

    return (
        <ul
            style={{
                display: 'flex',
                gap: 'var(--spectrum-global-dimension-size-100)',
                flexDirection: 'column',
            }}
        >
            {robot.cameras?.map((camera, idx) => {
                return (
                    <li key={idx}>
                        <CameraThumbnail name={camera.name} fingerprint={camera.fingerprint} />
                    </li>
                );
            })}
        </ul>
    );
};
