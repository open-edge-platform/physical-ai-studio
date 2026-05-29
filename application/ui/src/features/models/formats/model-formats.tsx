import { Suspense } from 'react';

import {
    Badge,
    Button,
    Content,
    ContextualHelp,
    Divider,
    Flex,
    Grid,
    Heading,
    Icon,
    Loading,
    Text,
    View,
} from '@geti-ui/ui';
import { DownloadIcon } from '@geti-ui/ui/icons';

import { $api, fetchClient } from '../../../api/client';
import type { components, SchemaModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, type InferenceBackendConfig } from '../inference-backends';

interface ModelExportsProps {
    model: SchemaModel;
}

type BackendExportDetail = components['schemas']['BackendExportDetail'];
type ExportBackend = components['schemas']['ExportBackend'];
type ModelDetailResponse = components['schemas']['ModelDetailResponse'];

const isExportBackend = (backendType: string): backendType is ExportBackend => backendType in INFERENCE_BACKENDS;
const cardGridColumns = 'repeat(auto-fill, minmax(min(100%, var(--spectrum-global-dimension-size-4600)), 1fr))';

const InferenceBackendLogo = ({ backend, isAvailable }: { backend: InferenceBackendConfig; isAvailable: boolean }) => {
    const Logo = backend.logo;
    const unavailableStyle = isAvailable ? undefined : { opacity: 0.4 };

    return (
        <Flex alignItems='center' gap='size-200' UNSAFE_style={unavailableStyle}>
            <Flex>
                <Logo height={'50px'} width={'50px'} />
            </Flex>
            <Flex direction='column' gap='size-10' justifyContent={'center'}>
                <Heading level={4} marginBottom={0}>
                    {backend.label}
                </Heading>
                <Text
                    UNSAFE_style={{
                        fontSize: '11px',
                    }}
                >
                    {backend.description}
                </Text>
            </Flex>
        </Flex>
    );
};

const Unavailable = ({ backend }: { backend: InferenceBackendConfig }) => {
    return (
        <Flex direction='column' gap='size-100' alignItems={'end'}>
            <Badge variant={'negative'} UNSAFE_style={{ padding: 0, opacity: 0.9 }}>
                Unavailable
            </Badge>
            <ContextualHelp variant='help'>
                <Heading>Export missing</Heading>
                <Content>
                    <Text>
                        This model does not include an exported model for {backend.label}. Try retraining the model to
                        restart the model export.
                    </Text>
                </Content>
            </ContextualHelp>
        </Flex>
    );
};

const formatSize = (bytes: number): string => {
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
        return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(1)} MB`;
};

const ModelFormatSize = ({ exportDetail }: { exportDetail: BackendExportDetail | undefined }) => {
    if (exportDetail === undefined) {
        return (
            <Flex direction='column' gap='size-10'>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>Size</Text>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>—</Text>
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap='size-10'>
            <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>Size</Text>
            <Text>{formatSize(exportDetail.size_bytes)}</Text>
        </Flex>
    );
};

const ModelPrecision = ({ exportDetail }: { exportDetail: BackendExportDetail | undefined }) => {
    if (exportDetail === undefined) {
        return (
            <Flex direction='column' gap='size-10'>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>Precision</Text>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>—</Text>
            </Flex>
        );
    }

    // At the moment we only support FP16 models
    return (
        <Flex direction='column' gap='size-10'>
            <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>Precision</Text>
            <Text>FP16</Text>
        </Flex>
    );
};

interface BackendCardProps {
    modelDetail: ModelDetailResponse;
    backendType: ExportBackend;
    model: SchemaModel;
}

const BackendCard = ({ modelDetail, backendType, model }: BackendCardProps) => {
    const exportDetail = modelDetail.exports.find(({ type }) => type === backendType);
    const isAvailable = exportDetail !== undefined;
    const backend = INFERENCE_BACKENDS[backendType];
    const downloadUrl = fetchClient.PATH('/api/models/{model_id}/exports/{backend}/download', {
        params: { path: { model_id: model.id!, backend: backendType } },
    });

    return (
        <View
            backgroundColor={'gray-50'}
            paddingY='size-200'
            borderRadius={'medium'}
            borderWidth='thin'
            borderColor={'gray-100'}
        >
            <Flex direction='column' justifyContent='space-between' gap='size-200'>
                <View paddingX='size-200'>
                    <Flex justifyContent={'space-between'}>
                        <Flex direction='column' gap='size-100' marginEnd='size-200' justifyContent='center'>
                            <InferenceBackendLogo backend={backend} isAvailable={isAvailable} />
                        </Flex>

                        {isAvailable === false && <Unavailable backend={backend} />}
                    </Flex>
                </View>

                <Divider size='S' />

                <View paddingX='size-200'>
                    <Flex gap='size-400' marginTop='size-100' width='100%'>
                        <ModelFormatSize exportDetail={exportDetail} />
                        <ModelPrecision exportDetail={exportDetail} />
                        {isAvailable && (
                            <View marginStart='auto' alignSelf={'center'}>
                                <Button
                                    href={downloadUrl}
                                    aria-label={`Download ${backend.label} export`}
                                    UNSAFE_style={{
                                        color: 'inherit',
                                        display: 'inline-flex',
                                        textDecoration: 'none',
                                        paddingInline: 'var(--spectrum-global-dimension-size-200)',
                                        alignItems: 'center',
                                    }}
                                    target='_blank'
                                    rel='noopener noreferrer'
                                    variant='secondary'
                                >
                                    <Icon marginEnd='size-100'>
                                        <DownloadIcon />
                                    </Icon>
                                    <span>Download</span>
                                </Button>
                            </View>
                        )}
                    </Flex>
                </View>
            </Flex>
        </View>
    );
};

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
