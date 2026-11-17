import { useState } from 'react';

import { Item, Picker } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaExportVariantDetail, SchemaModelDetailResponse } from '../../../api/openapi-spec';

export type ModelVariant = SchemaExportVariantDetail['id'];

export const useVariantSelection = (
    modelId: string | undefined,
    backend: string,
    initialVariant: ModelVariant = 'original',
    detail?: SchemaModelDetailResponse
) => {
    const query = $api.useQuery(
        'get',
        '/api/models/{model_id}',
        { params: { path: { model_id: modelId ?? '' } } },
        { enabled: modelId !== undefined && detail === undefined }
    );
    const key = `${modelId}/${backend}`;
    const [selection, setSelection] = useState<{ key: string; variant: ModelVariant }>();
    const variants = (detail ?? query.data)?.exports.find((item) => item.type === backend)?.variants ?? [];
    const requested = selection?.key === key ? selection.variant : initialVariant;
    const selected = variants.find((variant) => variant.id === requested) ?? variants[0];
    return {
        variants,
        selected,
        variant: selected?.id ?? 'original',
        isPending: detail === undefined && query.isPending,
        isError: detail === undefined && query.isError,
        setVariant: (variant: ModelVariant) => setSelection({ key, variant }),
    };
};

export const VariantSelection = ({
    variants,
    variant,
    onChange,
}: {
    variants: SchemaExportVariantDetail[];
    variant: ModelVariant;
    onChange: (variant: ModelVariant) => void;
}) => (
    <Picker
        label='Variant'
        items={variants}
        selectedKey={variant}
        onSelectionChange={(key) => {
            if (key === 'original' || key === 'int8') onChange(key);
        }}
    >
        {(item) => <Item key={item.id}>{item.id === 'int8' ? 'INT8' : 'Original'}</Item>}
    </Picker>
);
