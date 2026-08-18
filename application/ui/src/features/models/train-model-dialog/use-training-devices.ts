import { useMemo } from 'react';

import { $api } from '../../../api/client';
import { SchemaDeviceInfo } from '../../../api/openapi-spec';

/** Pick the device with the most VRAM (if any) from a list of reported devices. */
export const pickBestDevice = (devices: SchemaDeviceInfo[]): SchemaDeviceInfo | null =>
    devices
        .filter((d) => d.type !== 'cpu' && d.memory != null)
        .reduce((best: SchemaDeviceInfo | null, device) => {
            if (best === null || (device.memory ?? 0) > (best.memory ?? 0)) {
                return device;
            }

            return best;
        }, null);

/**
 * Reads the training devices endpoint and normalizes the response.
 *
 * The endpoint always reports this Studio host's local training devices. Remote
 * trainer configuration is selected independently for each submitted job.
 */
export const useTrainingDevices = () => {
    const { data } = $api.useQuery('get', '/api/system/devices/training', {}, { refetchOnMount: 'always' });

    return {
        devices: data?.devices ?? [],
    };
};

export const useBestTrainingDevice = (): SchemaDeviceInfo | null => {
    const { devices } = useTrainingDevices();

    return useMemo(() => pickBestDevice(devices), [devices]);
};
