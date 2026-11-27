import { SchemaPolicyTrainingConfiguration } from '../../api/openapi-spec';

export const getMockedPolicyTrainingConfiguration = (
    overrides: Partial<SchemaPolicyTrainingConfiguration> = {}
): SchemaPolicyTrainingConfiguration => ({
    defaults: {
        batch_size: 8,
        max_epochs: 5,
        num_workers: 'auto',
        precision: null,
        compile_model: false,
        auto_scale_batch_size: false,
        augment_images: false,
        validation: { split: 0.1 },
        lora: null,
        snapflow: null,
    },
    help: {},
    ...overrides,
});
