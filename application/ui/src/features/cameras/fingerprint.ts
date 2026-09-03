export type CameraFingerprint = Record<string, unknown>;

export const fingerprintKey = (fingerprint: CameraFingerprint | null | undefined): string | undefined => {
    if (!fingerprint) return undefined;
    const canonicalize = (value: unknown): unknown => {
        if (Array.isArray(value)) return value.map(canonicalize);
        if (value !== null && typeof value === 'object') {
            return Object.fromEntries(
                Object.entries(value)
                    .map(([key, item]): [string, unknown] => [key, canonicalize(item)])
                    .sort(([a], [b]) => a.localeCompare(b))
            );
        }
        return value;
    };

    return JSON.stringify(canonicalize(fingerprint));
};

export const formatFingerprint = (fingerprint: CameraFingerprint | null | undefined): string => {
    if (!fingerprint) return 'Camera needs reselection';
    const bus = typeof fingerprint.bus === 'string' && fingerprint.bus ? fingerprint.bus : null;
    const sensor = typeof fingerprint.sensor === 'string' && fingerprint.sensor ? fingerprint.sensor : null;
    const serial = typeof fingerprint.serial === 'string' && fingerprint.serial ? fingerprint.serial : null;
    const url = typeof fingerprint.url === 'string' && fingerprint.url ? fingerprint.url : null;
    const index = typeof fingerprint.index === 'number' ? fingerprint.index : null;

    const serial_bus = bus && serial ? `${serial} @ ${bus}` : serial;
    if (serial_bus) return serial_bus;

    if (url) return url;

    const bus_sensor = bus && sensor ? `${bus} - ${sensor}` : bus;
    if (bus_sensor?.includes('v4l2loopback')) {
        return index !== null ? `Virtual camera ${index}` : 'Virtual camera';
    }
    if (bus_sensor) return bus_sensor;
    if (index !== null) return `Camera ${index}`;

    return fingerprintKey(fingerprint) ?? 'Camera needs reselection';
};
