import { SchemaRemoteTrainer } from '../../../api/openapi-spec';

export const INSTANCE_TYPES = [
    {
        id: 'direct',
        label: 'Self Managed Remote Trainer',
        description: 'Connect to a trainer service you deploy and manage.',
    },
    {
        id: 'ssh',
        label: 'SSH Remote Trainer',
        description: 'Let Studio set up and manage a trainer over SSH.',
    },
    {
        id: 'aws_batch',
        label: 'AWS Provider',
        description: 'Run training jobs on on-demand GPU compute on AWS.',
    },
] as const;

export type InstanceType = (typeof INSTANCE_TYPES)[number]['id'];

export const instanceTypeLabel = (type: InstanceType): string => INSTANCE_TYPES.find(({ id }) => id === type)!.label;

export type TrainingTargetRow = { kind: 'direct-url'; trainer: SchemaRemoteTrainer };

export const trainingTargetRowId = (row: TrainingTargetRow): string => row.trainer.id;

export const trainingTargetRowName = (row: TrainingTargetRow): string => row.trainer.name;
