import { Flex, Grid, View } from '@geti/ui';

import { OverviewStep } from '../../features/robots/calibration/steps/overview-step';

export const Calibration = () => {
    return (
        <View paddingY='size-400'>
            <Flex direction='column' gap='size-200' height='100%' maxHeight='100vh'>
                <Grid
                    areas={['controls controls', 'table table', 'navigation navigation', 'controller calibration']}
                    height='100%'
                    rows={['auto', 'auto', 'auto', '1fr']}
                    columns={['2fr', '1fr']}
                    gap='size-400'
                >
                    <OverviewStep />
                </Grid>
            </Flex>
        </View>
    );
};
