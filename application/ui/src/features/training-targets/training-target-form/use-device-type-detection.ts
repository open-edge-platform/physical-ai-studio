import { $api } from '../../../api/client';
import { SchemaDeviceTypeDetection } from '../../../api/openapi-spec';

// Mirrors the SSH-provisioned subset of `DeviceType` the form's Picker offers.
// `detect_device_type` never reports "cpu" or "npu" - only CUDA/XPU hosts have a
// trainer image - but the generated type is the full backend enum, so this
// narrows it defensively rather than trusting the server to never surprise us.
const DETECTABLE_DEVICE_TYPES = ['cuda', 'xpu'] as const;
type DetectableDeviceType = (typeof DETECTABLE_DEVICE_TYPES)[number];

const isDetectableDeviceType = (value: string): value is DetectableDeviceType =>
    (DETECTABLE_DEVICE_TYPES as readonly string[]).includes(value);

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
        detectedDeviceType: detected && isDetectableDeviceType(detected) ? detected : undefined,
        isDetecting: isFetching,
    };
};
