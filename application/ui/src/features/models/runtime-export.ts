export const runtimeExportUrl = ({
    modelId,
    environmentId,
    backend,
    device,
    task,
}: {
    modelId: string;
    environmentId: string;
    backend: string;
    device: string;
    task?: string;
}): string => {
    const params = new URLSearchParams({
        environment_id: environmentId,
        device,
    });
    if (task !== undefined && task.trim() !== '') {
        params.set('task', task);
    }
    return `/api/models/${modelId}/exports/${backend}/download?${params.toString()}`;
};
