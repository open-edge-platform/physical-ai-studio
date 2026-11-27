import { useEffect, useMemo, useState } from 'react';

import { $api } from '../../../api/client';
import type { SchemaExportBackendOutput, SchemaRemoteTrainJobPayload } from '../../../api/openapi-spec';
import { getSupportedInferenceDevices } from '../backend-selection/backend-selection';
import { INFERENCE_BACKENDS, InferenceBackendConfig, isExportBackend } from '../inference-backends';

export type ExecuTorchDelegate = SchemaRemoteTrainJobPayload['executorch_delegate'];

export interface ExportSelection {
    /** Export formats this policy can produce, in the order the backend reports them. */
    backends: InferenceBackendConfig[];
    /** Formats to export after training, as the train payload takes them. */
    selectedBackends: SchemaExportBackendOutput[];
    setSelectedBackends: (backends: SchemaExportBackendOutput[]) => void;
    executorchDelegate: ExecuTorchDelegate;
    setExecutorchDelegate: (delegate: ExecuTorchDelegate) => void;
    /** OpenVINO device compiled into the ExecuTorch program; fixed once exported. */
    executorchDevice: string;
    setExecutorchDevice: (device: string) => void;
    /** OpenVINO devices the training machine can compile for. */
    executorchDevices: { device: string; name: string }[];
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
export const useExportBackends = (policy: string, remoteTrainerId?: string): ExportSelection => {
    const {
        data: backendsByPolicy,
        isLoading,
        isError,
    } = $api.useQuery(
        'get',
        '/api/policies/backends',
        { params: { query: { remote_trainer_id: remoteTrainerId } } },
        { staleTime: remoteTrainerId ? 0 : undefined }
    );

    const backends = useMemo(
        () => (backendsByPolicy?.[policy] ?? []).filter(isExportBackend).map((backend) => INFERENCE_BACKENDS[backend]),
        [backendsByPolicy, policy]
    );

    const [selectedBackends, setSelectedBackends] = useState<SchemaExportBackendOutput[]>([]);
    const [executorchDelegate, setExecutorchDelegate] = useState<ExecuTorchDelegate>('openvino');
    const [executorchDevice, setExecutorchDevice] = useState('CPU');

    // The OpenVINO device is compiled in at export, which runs on the training
    // machine. Local devices are known; a remote trainer's are not, so offer the
    // generic ones OpenVINO resolves there.
    const { data: inferenceDevices = [] } = $api.useQuery('get', '/api/system/devices/inference', undefined, {
        enabled: remoteTrainerId === undefined,
    });
    const executorchDevices =
        remoteTrainerId === undefined
            ? getSupportedInferenceDevices(inferenceDevices, 'openvino')
            : [
                  { device: 'CPU', name: 'CPU' },
                  { device: 'GPU', name: 'GPU' },
              ];

    // Every supported format is exported by default, which is what training did
    // before the formats could be chosen at all. A policy switch changes the
    // available formats, so the selection starts over with it.
    const availableBackends = backends.map((backend) => backend.type).join(' ');
    useEffect(() => {
        setSelectedBackends(availableBackends === '' ? [] : availableBackends.split(' ').filter(isExportBackend));
    }, [availableBackends]);

    const error = isError
        ? 'Cannot verify export formats on this training target.'
        : !isLoading && backends.length > 0 && selectedBackends.length === 0
          ? 'Pick at least one export format; a model with no exports can only be retrained, not deployed.'
          : null;

    return {
        backends,
        selectedBackends,
        setSelectedBackends,
        executorchDelegate,
        setExecutorchDelegate,
        executorchDevice,
        setExecutorchDevice,
        executorchDevices,
        isLoading,
        error,
    };
};
