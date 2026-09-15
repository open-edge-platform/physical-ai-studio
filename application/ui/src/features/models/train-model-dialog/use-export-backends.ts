import { useEffect, useMemo, useState } from 'react';

import { $api } from '../../../api/client';
import type { SchemaExportBackend } from '../../../api/openapi-spec';
import { INFERENCE_BACKENDS, InferenceBackendConfig, isExportBackend } from '../inference-backends';

export interface ExportSelection {
    /** Export formats this policy can produce, in the order the backend reports them. */
    backends: InferenceBackendConfig[];
    /** Formats to export after training, as the train payload takes them. */
    selectedBackends: SchemaExportBackend[];
    setSelectedBackends: (backends: SchemaExportBackend[]) => void;
    isLoading: boolean;
    /** Why the selection can't be trained with, or null when it is fine. */
    error: string | null;
}

/**
 * The export formats a policy supports, and which of them to produce.
 *
 * Not every policy can be exported to every format — ACT traces to ONNX and
 * ExecuTorch, the VLA policies only to Torch and OpenVINO — so the list comes
 * from the backend rather than being hardcoded here.
 */
export const useExportBackends = (policy: string): ExportSelection => {
    const { data: backendsByPolicy, isLoading } = $api.useQuery('get', '/api/policies/backends');

    const backends = useMemo(
        () => (backendsByPolicy?.[policy] ?? []).filter(isExportBackend).map((backend) => INFERENCE_BACKENDS[backend]),
        [backendsByPolicy, policy]
    );

    const [selectedBackends, setSelectedBackends] = useState<SchemaExportBackend[]>([]);

    // Every supported format is exported by default, which is what training did
    // before the formats could be chosen at all. A policy switch changes the
    // available formats, so the selection starts over with it.
    const availableBackends = backends.map((backend) => backend.type).join(' ');
    useEffect(() => {
        setSelectedBackends(availableBackends === '' ? [] : availableBackends.split(' ').filter(isExportBackend));
    }, [availableBackends]);

    const error =
        !isLoading && backends.length > 0 && selectedBackends.length === 0
            ? 'Pick at least one export format; a model with no exports can only be retrained, not deployed.'
            : null;

    return { backends, selectedBackends, setSelectedBackends, isLoading, error };
};
