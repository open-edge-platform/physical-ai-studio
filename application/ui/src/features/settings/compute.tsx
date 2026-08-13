import { Suspense } from 'react';

import { Loading, View } from '@geti-ui/ui';

import { RemoteTrainersPage } from '../remote-trainers/remote-trainers-page';

export const Compute = () => {
    return (
        <View height={'100%'} minHeight={0}>
            <Suspense fallback={<Loading />}>
                <RemoteTrainersPage />
            </Suspense>
        </View>
    );
};
