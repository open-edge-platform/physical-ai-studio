import { Grid, View } from '@geti/ui';

import { JointControls } from '../../features/robots/controller/joint-controls';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';

export const Controller = () => {
    // NOTE: this route should be disabled if the robot hasn't been configured yet
    return (
        <RobotModelsProvider>
            <Grid
                gap='size-200'
                UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-100)' }}
                areas={['controller controller', 'controls controls']}
                rows={['auto', 'min-content']}
                height='100%'
                minHeight={0}
            >
                <View gridArea='controller'>
                    <RobotViewer />
                </View>
                <JointControls />
            </Grid>
        </RobotModelsProvider>
    );
};
