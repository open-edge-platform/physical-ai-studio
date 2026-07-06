import { Suspense } from 'react';

import { Flex, Grid, Loading, minmax, View } from '@geti-ui/ui';

import { RobotForm } from '../../features/robots/robot-form/form';
import { Preview } from '../../features/robots/robot-form/preview';
import { SubmitNewRobotButton } from '../../features/robots/robot-form/submit-new-robot-button';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const New = () => {
    return (
        <Grid areas={['robot controls']} columns={[minmax('size-6000', 'auto'), '1fr']} height={'100%'}>
            <View gridArea='robot' backgroundColor={'gray-100'} padding='size-400'>
                <Suspense fallback={<CenteredLoading />}>
                    <RobotForm submitButton={<SubmitNewRobotButton />} />
                </Suspense>
            </View>
            <View gridArea='controls'>
                <Preview />
            </View>
        </Grid>
    );
};
