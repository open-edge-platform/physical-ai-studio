import { Grid, View } from '@geti/ui';

import { ActionProvider } from '../../features/robots/action-context';
import { Cameras } from '../../features/robots/controller/cameras';
import { JointControls } from '../../features/robots/controller/joint-controls';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';

export const Controller = () => {
    // NOTE: this route should be disabled if the robot hasn't been configured yet
    return (
        <ActionProvider>
            <Grid
                gap='size-200'
                UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-100)' }}
                areas={['cameras controller', 'controls controls']}
                height='100%'
            >
                <View gridArea='controller'>
                    <RobotViewer />
                </View>
                <View gridArea='cameras' alignSelf={'center'}>
                    <Cameras />
                </View>
                <JointControls />
            </Grid>
        </ActionProvider>
    );
};
