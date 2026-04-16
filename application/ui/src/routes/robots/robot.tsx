import { useState } from 'react';

import { Button, ButtonGroup, Flex, Grid, View } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { JointControls } from '../../features/robots/controller/joint-controls';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { useRobot } from '../../features/robots/use-robot';

export const Robot = () => {
    const robot = useRobot();

    const identifyMutation = $api.useMutation('post', '/api/hardware/identify', {
        meta: { skipInvalidation: true },
    });

    const onIdentify = identifyMutation.isPending
        ? undefined
        : () => {
              identifyMutation.mutate({ body: robot });
          };

    const [isConnected, setIsConnected] = useState(false);

    return (
        <View padding='size-400' height='100%' minHeight='0'>
            <RobotModelsProvider>
                <Grid
                    gap='size-200'
                    UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-100)' }}
                    areas={['actions', 'robot-viewer', 'controls ']}
                    rows={['auto', '1fr', 'min-content']}
                    height='100%'
                    maxHeight={'100vh'}
                    maxWidth='100%'
                    minHeight={0}
                    minWidth={0}
                >
                    <View gridArea='actions'>
                        <Flex justifyContent={'end'}>
                            <ButtonGroup>
                                <Button variant='secondary' onPress={onIdentify}>
                                    Identify
                                </Button>
                            </ButtonGroup>
                        </Flex>
                    </View>
                    <View
                        gridArea='robot-viewer'
                        overflow='auto'
                        minHeight={0}
                        UNSAFE_style={
                            isConnected
                                ? undefined
                                : {
                                      filter: 'grayscale(0.8)',
                                      opacity: 0.5,
                                  }
                        }
                    >
                        <RobotViewer robot={robot} />
                    </View>
                    <JointControls isConnected={isConnected} setIsConnected={setIsConnected} />
                </Grid>
            </RobotModelsProvider>
        </View>
    );
};
