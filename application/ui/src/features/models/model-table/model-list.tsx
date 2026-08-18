import { $api } from '../../../api/client';
import { SchemaTrainJob as SchemaJob, SchemaModel } from '../../../api/openapi-spec';
import { Table, TableColumn } from '../../../components/table/table';
import { useProjectId } from '../../projects/use-project';
import { ModelRow } from './model-table';

const MODEL_COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: '2fr', header: 'Model name' },
    { width: '1fr', header: 'Trained' },
    { width: '1fr', header: 'Duration' },
    { width: '1fr', header: 'Architecture' },
    { width: '1fr' },
    { width: 'auto', align: 'end' },
];

interface ModelListProps {
    models: SchemaModel[];
    jobs: SchemaJob[];
    onRetrain: (model: SchemaModel) => void;
    onViewLogs: (model: SchemaModel) => void;
}

export const ModelList = ({ models, jobs, onRetrain, onViewLogs }: ModelListProps) => {
    const sortedModels = models.toSorted(
        (a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime()
    );

    const jobsById = new Map(jobs.map((j) => [j.id, j]));

    const { project_id } = useProjectId();
    const deleteModelMutation = $api.useMutation('delete', '/api/models/{model_id}', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/models', { params: { path: { project_id } } }]],
        },
    });

    const deleteModel = (model: SchemaModel) => {
        deleteModelMutation.mutate({ params: { path: { model_id: model.id! } } });
    };

    return (
        <Table columns={MODEL_COLUMNS}>
            {sortedModels.map((model) => (
                <ModelRow
                    key={model.id}
                    model={model}
                    trainingJob={model.train_job_id ? jobsById.get(model.train_job_id) : undefined}
                    onDelete={() => deleteModel(model)}
                    onRetrain={() => onRetrain(model)}
                    onViewLogs={() => onViewLogs(model)}
                />
            ))}
        </Table>
    );
};
