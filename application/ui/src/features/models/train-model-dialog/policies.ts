const GiB = 1024 ** 3;

/** Show whole-GB capacity labels for the device and policy cards. */
export const formatBytes = (bytes: number): string => `${Math.ceil(bytes / GiB)} GB`;
