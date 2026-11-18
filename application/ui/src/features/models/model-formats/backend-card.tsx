import { type ReactNode } from 'react';

import { Button, DialogTrigger, Divider, Flex, Heading, Icon, Text, View } from '@geti-ui/ui';
import { DownloadIcon } from '@geti-ui/ui/icons';

import { fetchClient } from '../../../api/client';
import type { components, SchemaModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, type InferenceBackendConfig } from '../inference-backends';
import { RemoveBackendDialog } from './remove-backend-dialog';
import { RuntimeExportDialog } from './runtime-export-dialog';

import classes from './model-formats.module.css';

type BackendExportDetail = components['schemas']['BackendExportDetail'];
type ExportBackend = components['schemas']['physicalai__export__backends__ExportBackend'];
type ModelDetailResponse = components['schemas']['ModelDetailResponse'];

export const InferenceBackendLogo = ({
    backend,
    isAvailable,
}: {
    backend: InferenceBackendConfig;
    isAvailable: boolean;
}) => {
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
    children?: ReactNode;
    isExporting?: boolean;
}

export const BackendCard = ({ modelDetail, backendType, model, children, isExporting = false }: BackendCardProps) => {
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
            borderColor={isExporting ? 'transparent' : 'gray-100'}
            height='100%'
            UNSAFE_className={isExporting ? classes.exportingCard : undefined}
        >
            {isExporting && (
                <svg className={classes.exportBorder} aria-hidden='true' focusable='false'>
                    <rect x='1' y='1' rx='8' />
                </svg>
            )}
            <Flex direction='column' gap='size-200' height='100%'>
                <View paddingX='size-200' UNSAFE_className={isExporting ? classes.exportingIdentity : undefined}>
                    <Flex justifyContent={'space-between'}>
                        <Flex direction='column' gap='size-100' marginEnd='size-200' justifyContent='center'>
                            <InferenceBackendLogo backend={backend} isAvailable={isAvailable || isExporting} />
                        </Flex>
                    </Flex>
                </View>

                <Divider size='S' />
                {children && <View paddingX='size-200'>{children}</View>}

                {!isExporting && (
                    <View
                        paddingX='size-200'
                        marginTop='auto'
                        UNSAFE_style={isAvailable ? undefined : { opacity: 0.4 }}
                    >
                        <Flex gap='size-200' wrap='wrap' alignItems='end' marginTop='size-100' width='100%'>
                            <Flex gap='size-400' flexShrink={0}>
                                <ModelFormatSize exportDetail={exportDetail} />
                                <ModelPrecision exportDetail={exportDetail} />
                            </Flex>
                            {isAvailable && (
                                <View marginStart='auto' maxWidth='100%' flexShrink={0}>
                                    <Flex direction='column' gap='size-100'>
                                        {backendType === 'openvino' && (
                                            <DialogTrigger>
                                                <Button
                                                    aria-label={`Download ${backend.label} runtime export`}
                                                    variant='secondary'
                                                    UNSAFE_style={{ whiteSpace: 'nowrap' }}
                                                >
                                                    Runtime export
                                                </Button>
                                                {(close) => (
                                                    <RuntimeExportDialog
                                                        close={close}
                                                        model={model}
                                                        backend={backendType}
                                                    />
                                                )}
                                            </DialogTrigger>
                                        )}
                                        <Button
                                            href={downloadUrl}
                                            aria-label={`Download ${backend.label} export`}
                                            UNSAFE_style={{
                                                color: 'inherit',
                                                display: 'inline-flex',
                                                textDecoration: 'none',
                                                paddingInline: 'var(--spectrum-global-dimension-size-200)',
                                                alignItems: 'center',
                                                whiteSpace: 'nowrap',
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
                                        <DialogTrigger>
                                            <Button variant='negative'>Remove model</Button>
                                            {(close) => (
                                                <RemoveBackendDialog model={model} backend={backendType} close={close} />
                                            )}
                                        </DialogTrigger>
                                    </Flex>
                                </View>
                            )}
                        </Flex>
                    </View>
                )}
            </Flex>
        </View>
    );
};
