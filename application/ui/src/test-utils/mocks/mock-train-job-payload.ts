// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    SchemaLocalTrainJobPayloadOutput,
    SchemaRemoteTrainJobPayloadOutput,
    SchemaSshTrainJobPayloadOutput,
    SchemaTrainJob,
} from '../../api/openapi-spec';

type TrainJobPayload = SchemaTrainJob['payload'];

const basePayload = {
    project_id: 'project-1',
    dataset_id: 'dataset-1',
    policy: 'act',
    model_name: 'pick-and-place',
    batch_size: 8,
    num_workers: 'auto' as const,
    auto_scale_batch_size: false,
    val_split: 0.1,
    precision: 'bf16-mixed' as const,
    compile_model: false,
};

type BaseKeys = keyof typeof basePayload;

/**
 * Overrides for one payload variant: fields already covered by `basePayload` stay
 * optional, everything else (including the variant's own required fields, e.g.
 * `remote_trainer_id`) keeps the required/optional-ness declared by the schema.
 * This means a newly-added required field on a variant becomes a compile error
 * here automatically, with no manual overload edits needed.
 */
type VariantOverrides<T> = Partial<Pick<T, Extract<keyof T, BaseKeys>>> & Omit<T, BaseKeys>;

export function getMockedTrainJobPayload(
    overrides?: Partial<VariantOverrides<SchemaLocalTrainJobPayloadOutput>>
): SchemaLocalTrainJobPayloadOutput;
export function getMockedTrainJobPayload(
    overrides: VariantOverrides<SchemaRemoteTrainJobPayloadOutput>
): SchemaRemoteTrainJobPayloadOutput;
export function getMockedTrainJobPayload(
    overrides: VariantOverrides<SchemaSshTrainJobPayloadOutput>
): SchemaSshTrainJobPayloadOutput;
export function getMockedTrainJobPayload(overrides: Partial<TrainJobPayload> = {}): TrainJobPayload {
    return {
        ...basePayload,
        training_target: 'local',
        ...overrides,
    } as TrainJobPayload;
}
