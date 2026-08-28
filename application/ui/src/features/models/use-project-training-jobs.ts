import { $api } from '../../api/client';
import { SchemaTrainJob } from './train-model-dialog/train-model-dialog';

export const useProjectTrainingJobs = (project_id: string): SchemaTrainJob[] => {
    const { data: allJobs = [] } = $api.useQuery('get', '/api/jobs');

    return allJobs
        .filter((job) => job.project_id === project_id)
        .filter((job): job is SchemaTrainJob => job.type === 'training');
};
