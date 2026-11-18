import { useState, type ReactNode } from 'react';

import {
    ActionButton,
    Button,
    DialogContainer,
    Divider,
    Flex,
    Heading,
    Item,
    Menu,
    MenuTrigger,
    Text,
    View,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { fetchClient } from '../../../api/client';
import { SchemaExportVariantDetail, SchemaModel, SchemaModelExportAction } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, InferenceBackendConfig } from '../inference-backends';
import { RemoveModelDialog } from './remove-model-dialog';
import { RunVariant } from './run-variant';
import { RuntimeExportDialog } from './runtime-export-dialog';

import classes from './model-formats.module.css';

export const InferenceBackendLogo = ({
    backend,
    isAvailable,
    compact = false,
    variantLabel,
}: {
    backend: InferenceBackendConfig;
    isAvailable: boolean;
    compact?: boolean;
    variantLabel?: string;
}) => {
    const Logo = backend.logo;
    return (
        <Flex alignItems='center' gap='size-200' UNSAFE_style={isAvailable ? undefined : { opacity: 0.4 }}>
            <Logo height={compact ? '32px' : '50px'} width={compact ? '32px' : '50px'} />
            <Flex direction='column' gap='size-50'>
                <Flex alignItems='baseline' gap='size-100' wrap>
                    <Heading level={4} margin={0}>
                        {backend.label}
                    </Heading>
                    {variantLabel && <Text UNSAFE_className={classes.variantDescription}>{variantLabel}</Text>}
                </Flex>
                {!compact && <Text UNSAFE_style={{ fontSize: '11px' }}>{backend.description}</Text>}
            </Flex>
        </Flex>
    );
};

export const BackendCard = ({
    backendType,
    model,
    variant,
    variantId = variant?.id ?? 'original',
    isExporting = false,
    children,
    action,
    isCreateDisabled = false,
    onCreate,
}: {
    backendType: SchemaModelExportAction['backend'];
    model: SchemaModel;
    variant?: SchemaExportVariantDetail;
    variantId?: SchemaExportVariantDetail['id'];
    isExporting?: boolean;
    children?: ReactNode;
    action?: SchemaModelExportAction;
    isCreateDisabled?: boolean;
    onCreate: (action: SchemaModelExportAction) => void;
}) => {
    const [dialog, setDialog] = useState<'runtime' | 'remove' | null>(null);
    const backend = INFERENCE_BACKENDS[backendType];
    const label = variantId === 'int8' ? 'INT8' : 'Original';
    const megabytes = (variant?.size_bytes ?? 0) / 1024 ** 2;
    const size = megabytes >= 1024 ? `${(megabytes / 1024).toFixed(1)} GB` : `${megabytes.toFixed(1)} MB`;
    const weights = variant?.weight_compression === 'int8_sym' ? 'INT8 symmetric' : 'Original';
    const downloadUrl = fetchClient.PATH('/api/models/{model_id}/exports/{backend}/download', {
        params: { path: { model_id: model.id!, backend: backendType }, query: { variant: variantId } },
    });
    return (
        <section
            aria-label={variantId === 'original' ? `${backend.label} model format` : `${backend.label} · ${label}`}
            aria-busy={isExporting}
        >
            <View
                backgroundColor='gray-50'
                padding='size-200'
                borderRadius='medium'
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
                <Flex direction='column' justifyContent='space-between' gap='size-200' height='100%'>
                    <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                        <View UNSAFE_className={isExporting ? classes.exportingIdentity : undefined}>
                            <InferenceBackendLogo
                                backend={backend}
                                isAvailable={variant !== undefined || isExporting}
                                compact
                                variantLabel={variantId === 'int8' ? label : undefined}
                            />
                        </View>
                        {variant && (
                            <MenuTrigger>
                                <ActionButton isQuiet aria-label={`Actions for ${backend.label} ${label}`}>
                                    <MoreMenu />
                                </ActionButton>
                                <Menu
                                    onAction={(key) => {
                                        if (key === 'runtime' || key === 'remove') setDialog(key);
                                    }}
                                >
                                    <Item key='download' href={downloadUrl} target='_blank' rel='noopener noreferrer'>
                                        Download
                                    </Item>
                                    {backendType === 'openvino' ? <Item key='runtime'>Runtime export</Item> : null}
                                    <Item key='remove'>Remove model</Item>
                                </Menu>
                            </MenuTrigger>
                        )}
                    </Flex>
                    <Divider size='S' />
                    {children}
                    {!isExporting && (
                        <Flex alignItems='end' gap='size-200' wrap>
                            <Flex direction='column'>
                                <Text>Size</Text>
                                <Text>{variant ? size : '—'}</Text>
                            </Flex>
                            <Flex direction='column'>
                                <Text>Weights</Text>
                                <Text>{variant ? weights : '—'}</Text>
                            </Flex>
                            <View marginStart='auto'>
                                <Flex gap='size-100' alignItems='center'>
                                    {action && (
                                        <Button
                                            variant='secondary'
                                            isDisabled={isCreateDisabled}
                                            onPress={() => onCreate(action)}
                                        >
                                            Optimize
                                        </Button>
                                    )}
                                    {variant && <RunVariant model={model} backend={backendType} variant={variantId} />}
                                </Flex>
                            </View>
                        </Flex>
                    )}
                </Flex>
            </View>
            <DialogContainer onDismiss={() => setDialog(null)}>
                {dialog === 'runtime' && variant && (
                    <RuntimeExportDialog
                        model={model}
                        backend={backendType}
                        variant={variantId}
                        close={() => setDialog(null)}
                    />
                )}
                {dialog === 'remove' && variant && (
                    <RemoveModelDialog
                        model={model}
                        backend={backendType}
                        variant={variant}
                        close={() => setDialog(null)}
                    />
                )}
            </DialogContainer>
        </section>
    );
};
