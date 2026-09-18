const GB = 1024 ** 3;

/** Format bytes as a human-readable GB string. */
export const formatBytes = (bytes: number): string => {
    const gb = bytes / GB;
    return gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
};

/**
 * Available training policies with hardware requirements.
 *
 * `minVRAM` is the peak VRAM measured over one training step at batch_size=1.
 * Optimizer state drives the peak, so it tracks the trainable parameter count
 * and barely moves with batch size: Pi0.5 needs ~38 GB at both batch 1 and 8.
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
        description: 'Action Chunking with Transformers, lightweight and fast to train (BSD 3-Clause license)',
        minVRAM: 2 * GB,
    },
    {
        id: 'smolvla',
        name: 'SmolVLA',
        description: 'Small Vision-Language-Action model based on SmolVLM2-500M (Apache 2.0 license)',
        minVRAM: 3 * GB,
    },
    {
        id: 'pi05',
        name: 'Pi0.5',
        description: 'Flow-matching VLA with discrete state encoding and longer context (Gemma license)',
        minVRAM: 40 * GB,
    },
];
