import { Suspense } from 'react';

import { Grid, Loading } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import type { SchemaModel } from '../../../api/openapi-spec';
import { ModelInterface } from './model-interface';
import { TrainingConfiguration } from './training-configuration';

interface ModelDetailsProps {
    model: SchemaModel;
}

const ModelDetailsContent = ({ model }: { model: SchemaModel }) => {
    const { data: modelDetail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });

    return (
        <Grid
            areas={{
                L: ['model-interface training', 'model-interface training'],
                M: ['model-interface', 'training'],
            }}
            gap='size-200'
            columns={['auto 1fr']}
        >
            <ModelInterface exports={modelDetail.exports} />
            <TrainingConfiguration summary={modelDetail.training_summary} hparams={modelDetail.hparams} />
        </Grid>
    );
};

export const ModelDetails = ({ model }: ModelDetailsProps) => {
    return (
        <Suspense fallback={<Loading mode='inline' size='M' marginY='size-400' />}>
            <ModelDetailsContent model={model} />
        </Suspense>
    );
};
