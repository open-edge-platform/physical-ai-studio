const GB = 1024 ** 3;

/** Format bytes as a human-readable GB string. */
export const formatBytes = (bytes: number): string => {
    const gb = bytes / GB;
    return gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
};

/**
 * Available training policies with hardware requirements.
 *
 * `minVRAM` is the estimated minimum VRAM (in bytes) required to train with batch_size=1.
 */
export const MODELS: ReadonlyArray<{
    id: string;
    name: string;
    description: string;
    minVRAM: number;
}> = [
    {
        id: 'act',
        name: 'ACT',
        description: 'Action Chunking with Transformers, lightweight and fast to train',
        minVRAM: 2 * GB,
    },
    {
        id: 'smolvla',
        name: 'SmolVLA',
        description: 'Small Vision-Language-Action model based on SmolVLM2-500M',
        minVRAM: 8 * GB,
    },
    {
        id: 'pi05',
        name: 'Pi0.5',
        description: 'Enhanced Pi0 with discrete state encoding and longer context',
        minVRAM: 16 * GB,
    },
];
