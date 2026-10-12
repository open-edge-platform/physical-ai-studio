const GiB = 1024 ** 3;

/** Show whole-GB capacity labels for the device and policy cards. */
export const formatBytes = (bytes: number): string => `${Math.ceil(bytes / GiB)} GB`;

/**
 * Available training policies with hardware requirements.
 *
 * `minVRAM` is the peak VRAM measured over one training step at batch_size=1.
 * Optimizer state drives the peak, so it tracks the trainable parameter count
 * and barely moves with batch size: Pi0.5 needs ~38 GiB at both batch 1 and 8.
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
        minVRAM: 2 * GiB,
    },
    {
        id: 'smolvla',
        name: 'SmolVLA',
        description: 'Small Vision-Language-Action model based on SmolVLM2-500M (Apache 2.0 license)',
        minVRAM: 3 * GiB,
    },
    {
        id: 'molmoact2',
        name: 'MolmoAct2',
        description: 'Vision-Language-Action model with flow-matching action generation (Apache 2.0 license)',
        minVRAM: 80 * GiB,
    },
    {
        id: 'pi05',
        name: 'Pi0.5',
        description: 'Flow-matching VLA with discrete state encoding and longer context (Gemma license)',
        minVRAM: 40 * GiB,
    },
    {
        id: 'rldx1',
        name: 'RLDX-1',
        description: 'General-purpose robot foundation model for dexterous manipulation (RLWRLD Model License v1.0)',
        minVRAM: 40 * GiB,
    },
    {
        id: 'xr0',
        name: 'XR0',
        description:
            'Vision-Language-Action model with a Qwen3-VL backbone and flow-matching action expert ' +
            '(Apache 2.0 license)',
        minVRAM: 60 * GiB,
    },
];
