import { Suspense } from 'react';

import { Grid, Loading } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import type { components, SchemaModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS } from '../inference-backends';
import { BackendCard } from './backend-card';

interface ModelExportsProps {
    model: SchemaModel;
}

type ExportBackend = components['schemas']['ExportBackend'];

const isExportBackend = (backendType: string): backendType is ExportBackend => backendType in INFERENCE_BACKENDS;

const cardGridColumns = 'repeat(auto-fill, minmax(min(100%, var(--spectrum-global-dimension-size-4600)), 1fr))';

const ModelFormatsContents = ({ model }: { model: SchemaModel }) => {
    const { data: modelDetail } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id: model.id! } },
    });
    const { data: policyBackends } = $api.useSuspenseQuery('get', '/api/policies/backends');

    const backends = (policyBackends[model.policy] ?? []).filter(isExportBackend);

    return (
        <Grid
            gap='size-200'
            marginTop='size-400'
            UNSAFE_style={{
                gridTemplateColumns: cardGridColumns,
            }}
        >
            {backends.map((backendType) => {
                return (
                    <BackendCard key={backendType} backendType={backendType} model={model} modelDetail={modelDetail} />
                );
            })}
        </Grid>
    );
};

export const ModelFormats = ({ model }: ModelExportsProps) => {
    return (
        <Suspense fallback={<Loading mode='inline' size='M' marginY='size-400' />}>
            <ModelFormatsContents model={model} />
        </Suspense>
    );
};
