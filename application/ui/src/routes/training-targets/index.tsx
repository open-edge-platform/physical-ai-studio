import { Suspense } from 'react';

import { Flex, Loading } from '@geti-ui/ui';

import { TrainingTargetsPage } from '../../features/training-targets/training-targets-page';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

export const Index = () => {
    return (
        <Suspense fallback={<CenteredLoading />}>
            <TrainingTargetsPage />
        </Suspense>
    );
};
