import {
    Badge,
    Button,
    Content,
    ContextualHelp,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Icon,
    Text,
    View,
} from '@geti-ui/ui';
import { DownloadIcon } from '@geti-ui/ui/icons';

import { fetchClient } from '../../../api/client';
import type { components, SchemaModel } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, type InferenceBackendConfig } from '../inference-backends';
import { OptimizeModel } from './optimize-model';
import { RuntimeExportDialog } from './runtime-export-dialog';
import { useVariantSelection, VariantSelection } from './variant-selection';

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

export const Unavailable = ({ backend }: { backend: InferenceBackendConfig }) => {
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

const ModelFormatSize = ({ exportDetail }: { exportDetail: Pick<BackendExportDetail, 'size_bytes'> | undefined }) => {
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

const ModelPrecision = ({ isAvailable, isInt8 }: { isAvailable: boolean; isInt8: boolean }) => {
    if (!isAvailable) {
        return (
            <Flex direction='column' gap='size-10'>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>Precision</Text>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>—</Text>
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap='size-10'>
            <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>Precision</Text>
            <Text>{isInt8 ? 'INT8 (weights)' : 'FP16'}</Text>
        </Flex>
    );
};

interface BackendCardProps {
    modelDetail: ModelDetailResponse;
    backendType: ExportBackend;
    model: SchemaModel;
}

export const BackendCard = ({ modelDetail, backendType, model }: BackendCardProps) => {
    const backendDetail = modelDetail.exports.find(({ type }) => type === backendType);
    const selection = useVariantSelection(model.id, backendType, 'original', modelDetail);
    const exportDetail = selection.selected ?? backendDetail;
    const isAvailable = backendDetail !== undefined;
    const backend = INFERENCE_BACKENDS[backendType];
    const downloadUrl = fetchClient.PATH('/api/models/{model_id}/exports/{backend}/download', {
        params: { path: { model_id: model.id!, backend: backendType }, query: { variant: selection.variant } },
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

                {selection.variants.length > 1 && (
                    <View paddingX='size-200'>
                        <VariantSelection
                            variants={selection.variants}
                            variant={selection.variant}
                            onChange={selection.setVariant}
                        />
                    </View>
                )}

                <View paddingX='size-200'>
                    <Flex gap='size-400' marginTop='size-100' width='100%'>
                        <ModelFormatSize exportDetail={exportDetail} />
                        <ModelPrecision
                            isAvailable={isAvailable}
                            isInt8={selection.selected?.weight_compression === 'int8_sym'}
                        />
                        {isAvailable && (
                            <View marginStart='auto' alignSelf={'center'}>
                                <Flex gap='size-100'>
                                    {backendType === 'openvino' && (
                                        <DialogTrigger>
                                            <Button
                                                aria-label={`Download ${backend.label} runtime export`}
                                                variant='secondary'
                                            >
                                                Runtime export
                                            </Button>
                                            {(close) => (
                                                <RuntimeExportDialog
                                                    close={close}
                                                    model={model}
                                                    backend={backendType}
                                                    variant={selection.variant}
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
                                </Flex>
                            </View>
                        )}
                    </Flex>
                </View>
                {isAvailable && backendType === 'openvino' && (
                    <View paddingX='size-200'>
                        <OptimizeModel
                            modelId={model.id!}
                            hasInt8={selection.variants.some((variant) => variant.id === 'int8')}
                        />
                    </View>
                )}
            </Flex>
        </View>
    );
};
