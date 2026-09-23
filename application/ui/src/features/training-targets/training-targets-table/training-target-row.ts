import { SchemaRemoteTrainer } from '../../../api/openapi-spec';

export type TrainingTargetRow = { kind: 'direct-url'; trainer: SchemaRemoteTrainer };

export const trainingTargetRowId = (row: TrainingTargetRow): string => row.trainer.id;

export const trainingTargetRowName = (row: TrainingTargetRow): string => row.trainer.name;
