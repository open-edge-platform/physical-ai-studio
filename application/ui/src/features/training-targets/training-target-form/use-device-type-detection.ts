import { $api } from '../../../api/client';
import { SchemaDeviceTypeDetection } from '../../../api/openapi-spec';
import { isSshDeviceType } from '../training-targets-table/remote-server-form/ssh-target-fields';

/**
 * Best-effort device-type autodetection for a selected SSH host alias.
 *
 * Fires once an alias is picked and stays disabled otherwise. A host the
 * backend could not probe (unreachable, no CUDA/XPU signal, etc.) comes back
 * as a 200 with `device_type: null` rather than an error, so callers only
 * need to check `detectedDeviceType` - there is nothing to distinguish
 * "still loading" from "detection failed" beyond `isDetecting`.
 */
export const useDeviceTypeDetection = (sshHostAlias: string | undefined) => {
    const { data, isFetching } = $api.useQuery(
        'get',
        '/api/remote-servers/aliases/{alias}/device-type',
        { params: { path: { alias: sshHostAlias ?? '' } } },
        { enabled: sshHostAlias !== undefined }
    );

    const detection: SchemaDeviceTypeDetection | undefined = data;
    const detected = detection?.device_type;

    return {
        detectedDeviceType: detected && isSshDeviceType(detected) ? detected : undefined,
        isDetecting: isFetching,
    };
};
