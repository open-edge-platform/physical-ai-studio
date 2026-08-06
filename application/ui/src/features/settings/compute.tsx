import { Suspense } from 'react';

import { Loading, View } from '@geti-ui/ui';

import { featureFlags } from '../../config/feature-flags';
import { RemoteTrainersPage } from '../remote-trainers/remote-trainers-page';

export const Compute = () => {
    return (
        <View height={'100%'} minHeight={0}>
            {featureFlags.remoteTrainers && (
                <Suspense fallback={<Loading />}>
                    <RemoteTrainersPage />
                </Suspense>
            )}
        </View>
    );
};
