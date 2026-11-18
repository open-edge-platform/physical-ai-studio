import { SchemaExportVariantDetail } from '../../api/openapi-spec';

export const runtimeExportUrl = ({
    modelId,
    environmentId,
    backend,
    device,
    task,
    variant = 'original',
}: {
    modelId: string;
    environmentId: string;
    backend: string;
    device: string;
    task?: string;
    variant?: SchemaExportVariantDetail['id'];
}): string => {
    const params = new URLSearchParams({
        environment_id: environmentId,
        device,
    });
    if (task !== undefined && task.trim() !== '') {
        params.set('task', task);
    }
    if (variant !== 'original') params.set('variant', variant);
    return `/api/models/${modelId}/exports/${backend}/download?${params.toString()}`;
};
