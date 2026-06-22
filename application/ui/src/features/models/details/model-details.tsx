import { Suspense, useState } from 'react';

import { Divider, Flex, Grid, Heading, Loading, Switch, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import type { components, SchemaModel, SchemaModelDetailResponse } from '../../../api/openapi-spec';
import { Box } from '../../../routes/models/box.component';

interface ModelDetailsProps {
    model: SchemaModel;
}

type ExportDetail = components['schemas']['BackendExportDetail'];
type IOFeature = components['schemas']['IOFeature'];

const SKIP_HPARAMS_KEYS = new Set(['dataset_stats']);

const isPrimitive = (v: unknown): v is string | number | boolean | null => {
    return v === null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean';
};

const DetailRow = ({ name, value }: { name: string; value: unknown }) => {
    const display = isPrimitive(value) ? String(value) : JSON.stringify(value);
    return (
        <>
            <Divider orientation='horizontal' size='S' />
            <View paddingY='size-50'>
                <Flex gap='size-200'>
                    <Text UNSAFE_style={{ flexShrink: 0, fontWeight: 'bold' }} width={'size-3000'}>
                        {name}
                    </Text>
                    <Text>{display}</Text>
                </Flex>
            </View>
        </>
    );
};

const formatShape = (shape: IOFeature['shape']) => {
    if (shape === null || shape === undefined) {
        return '-';
    }

    return shape.length === 0 ? 'scalar' : `[${shape.join(', ')}]`;
};

const getHeadingColor = (feature: IOFeature) => {
    if (feature.ftype === 'STATE') {
        return 'var(--coral)';
    }

    if (feature.ftype === 'VISUAL') {
        return 'var(--moss-tint-1)';
    }

    if (feature.ftype === 'LANGUAGE') {
        return 'var(--brand-daisy)';
    }

    if (feature.ftype === 'ACTION') {
        return 'var(--energy-blue)';
    }

    return undefined;
};

const FeatureRow = ({ feature }: { feature: IOFeature }) => {
    const color = getHeadingColor(feature);

    return (
        <View
            backgroundColor={'gray-100'}
            padding='size-100'
            borderRadius={'regular'}
            borderWidth={'thin'}
            borderColor='gray-300'
        >
            <Flex direction='column' gap='size-10'>
                <Text UNSAFE_style={{ fontWeight: 'bold', color }}>{feature.ftype}</Text>
                <Text UNSAFE_style={{ fontWeight: 'bold' }}>{feature.name}</Text>
                <Divider orientation='horizontal' size='S' marginY='size-50' />
                <View marginTop='size-50' UNSAFE_style={{ fontFamily: 'monospace' }}>
                    <Text marginEnd='size-100'>{feature.dtype}</Text>
                    <Text>{formatShape(feature.shape)}</Text>
                </View>
            </Flex>
        </View>
    );
};

const ModelInputInterface = ({ inputFeatures }: { inputFeatures: IOFeature[] }) => {
    return (
        <Flex direction='column' gap='size-100' width='size-3600'>
            <Text UNSAFE_style={{ fontWeight: 600 }}>Inputs ({inputFeatures.length})</Text>
            <Flex direction={'column'} gap='size-200'>
                {inputFeatures.map((feature) => {
                    return <FeatureRow key={feature.name} feature={feature} />;
                })}
            </Flex>
        </Flex>
    );
};

const ModelOutputInterface = ({ outputFeatures }: { outputFeatures: IOFeature[] }) => {
    return (
        <Flex direction='column' gap='size-100' width='size-3600'>
            <Text UNSAFE_style={{ fontWeight: 600 }}>Inputs ({outputFeatures.length})</Text>
            <Flex direction={'column'} gap='size-200'>
                {outputFeatures.map((feature) => {
                    return <FeatureRow key={feature.name} feature={feature} />;
                })}
            </Flex>
        </Flex>
    );
};

const ModelInterface = ({ exports }: { exports: ExportDetail[] }) => {
    const exportsWithIoSpec = exports.filter(({ io_spec }) => io_spec !== null && io_spec !== undefined);

    const mainFormat =
        exportsWithIoSpec.find((e) => {
            return e.type === 'torch';
        }) ?? exportsWithIoSpec.at(0);

    if (exportsWithIoSpec.length === 0 || mainFormat === undefined) {
        return null;
    }

    const inputFeatures = mainFormat.io_spec?.input_features ?? [];
    const outputFeatures = mainFormat.io_spec?.output_features ?? [];

    return (
        <View gridArea='model-interface'>
            <Box
                title='Model interface'
                content={
                    <Flex direction='row' gap='size-400'>
                        <ModelInputInterface inputFeatures={inputFeatures} />

                        <Divider size='S' orientation='vertical' />

                        <ModelOutputInterface outputFeatures={outputFeatures} />
                    </Flex>
                }
            />
        </View>
    );
};

const TrainingParameters = ({ summary }: { summary: SchemaModelDetailResponse['training_summary'] }) => {
    if (summary === undefined || summary === null) {
        return null;
    }

    return (
        <View>
            <Heading>Training parameters</Heading>
            <Flex direction='column' gap='size-75'>
                <DetailRow name='Max steps' value={summary.max_steps} />
                <DetailRow name='Batch size' value={summary.auto_scale_batch_size ? 'Auto' : summary.batch_size} />
                <DetailRow name='Workers' value={summary.num_workers ?? '—'} />
                {summary.precision && <DetailRow name='Precision' value={summary.precision} />}
                {summary.compile_model !== null && summary.compile_model !== undefined && (
                    <DetailRow name='Compiled' value={summary.compile_model ? 'Yes' : 'No'} />
                )}
                {summary.val_split !== null && summary.val_split !== undefined && summary.val_split > 0 && (
                    <DetailRow name='Validation split' value={summary.val_split} />
                )}
                {summary.device_type && <DetailRow name='Device' value={summary.device_type} />}
            </Flex>
        </View>
    );
};

const HyperParameters = ({ hparams }: { hparams: SchemaModelDetailResponse['training_summary'] }) => {
    const [showJSON, setShowJSON] = useState(false);

    if (hparams === null) {
        return null;
    }

    return (
        <View>
            <View marginBottom={'size-100'}>
                <Flex direction='row' justifyContent={'space-between'}>
                    <Heading marginTop='size-200'>Hyper parameters</Heading>
                    <Switch isSelected={showJSON} onChange={setShowJSON}>
                        Show JSON
                    </Switch>
                </Flex>
            </View>
            <View maxHeight='60vh' overflow='auto'>
                {showJSON ? (
                    <View backgroundColor={'gray-100'} borderWidth='thin' borderColor='gray-200' paddingX='size-100'>
                        <pre>{JSON.stringify(hparams, null, 4)}</pre>
                    </View>
                ) : (
                    <Flex direction='column' gap='size-75'>
                        {hparams &&
                            Object.entries(hparams)
                                .filter(([key]) => !SKIP_HPARAMS_KEYS.has(key))
                                .map(([key, value]) => <DetailRow key={key} name={key} value={value} />)}
                    </Flex>
                )}
            </View>
        </View>
    );
};

const TrainingConfiguration = ({
    summary,
    hparams,
}: {
    summary: SchemaModelDetailResponse['training_summary'];
    hparams: SchemaModelDetailResponse['training_summary'];
}) => {
    return (
        <View gridArea='training'>
            <Box
                title='Training configuration'
                content={
                    <Flex direction='column' gap='size-50'>
                        <TrainingParameters summary={summary} />

                        <HyperParameters hparams={hparams} />
                    </Flex>
                }
            />
        </View>
    );
};

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
