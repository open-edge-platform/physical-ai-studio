import { Suspense } from 'react';

import { Loading } from '@geti-ui/ui';

import { RecordingPage } from '../../../features/datasets/recording/recording-page';

export const Index = () => {
    return (
        <Suspense fallback={<Loading mode='overlay' />}>
            <RecordingPage />
        </Suspense>
    );
};
