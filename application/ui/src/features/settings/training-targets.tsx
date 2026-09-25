import { Suspense } from 'react';

import { Loading, View } from '@geti-ui/ui';

import { TrainingTargetsPage } from '../training-targets/training-targets-page';

export const TrainingTargets = () => {
    return (
        <View height={'100%'} minHeight={0}>
            <Suspense fallback={<Loading />}>
                <TrainingTargetsPage />
            </Suspense>
        </View>
    );
};
