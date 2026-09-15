import { SchemaRemoteServer, SchemaRemoteTrainer } from '../../../api/openapi-spec';

export type TrainingTargetRow =
    { kind: 'direct-url'; trainer: SchemaRemoteTrainer } | { kind: 'ssh'; server: SchemaRemoteServer };

export const trainingTargetRowId = (row: TrainingTargetRow): string =>
    row.kind === 'direct-url' ? row.trainer.id : row.server.id;

export const trainingTargetRowName = (row: TrainingTargetRow): string =>
    row.kind === 'direct-url' ? row.trainer.name : row.server.name;
