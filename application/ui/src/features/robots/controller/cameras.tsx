import { Grid } from '@geti/ui';

import RobotPicture from './../../../assets/robot-picture.png';

const Camera = ({ name }: { name: string }) => {
    return (
        <Grid
            UNSAFE_style={{
                background: 'var(--spectrum-global-color-gray-100)',
                border: '1px solid var(--spectrum-global-color-gray-200)',
                borderRadius: '8px',
                padding: 'var(--spectrum-global-dimension-size-150)',
            }}
            areas={['camera']}
        >
            <img
                src={RobotPicture}
                style={{
                    gridArea: 'camera',
                    borderRadius: '8px',
                }}
            />
            <div
                style={{
                    gridArea: 'camera',
                    display: 'flex',
                    alignItems: 'start',
                    justifyContent: 'end',
                }}
            >
                <span
                    style={{
                        background: 'var(--spectrum-global-color-gray-300)',
                        color: '#E3E3E5',
                        padding: 'var(--spectrum-global-dimension-size-50)',
                        borderRadius: '8px',
                        marginRight: '-4px',
                        marginTop: '-4px',
                        fontSize: '12px',
                        position: 'relative',
                    }}
                >
                    {name}
                </span>
            </div>
        </Grid>
    );
};

export const Cameras = () => {
    return (
        <ul
            style={{
                display: 'flex',
                gap: 'var(--spectrum-global-dimension-size-100)',
                flexDirection: 'column',
            }}
        >
            <li>
                <Camera name='Front Cam' />
            </li>
            <li>
                <Camera name='Grabber Cam' />
            </li>
            <li>
                <Camera name='Cam 3' />
            </li>
        </ul>
    );
};
