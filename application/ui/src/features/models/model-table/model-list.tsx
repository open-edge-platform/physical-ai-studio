import { View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaTrainJob as SchemaJob, SchemaModel } from '../../../api/openapi-spec';
import { useProjectId } from '../../projects/use-project';
import { ModelHeader, ModelRow } from './model-table';

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
        <View
            borderWidth='thin'
            borderColor={'gray-200'}
            borderBottomWidth='thin'
            borderBottomColor={'gray-200'}
            borderStartWidth='thin'
            borderStartColor={'gray-200'}
            borderEndWidth='thin'
            borderEndColor={'gray-200'}
        >
            <ModelHeader />
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
        </View>
    );
};
