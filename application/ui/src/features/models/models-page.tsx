import { useState } from 'react';

import { Button, DialogContainer, DialogTrigger, Flex, View } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { SchemaModel } from '../../api/openapi-spec';
import { LogsDialog } from '../logs/logs-dialog';
import { useProjectId } from '../projects/use-project';
import { JobList } from './job-table/job-list';
import { ModelList } from './model-table/model-list';
import { NoModelsPlaceholder } from './no-models-placeholder';
import { TrainModelDialog } from './train-model-dialog/train-model-dialog';
import { useJobUpdates } from './use-job-updates';
import { useProjectTrainingJobs } from './use-project-training-jobs';

export const ModelsPage = () => {
    const { project_id } = useProjectId();
    const { data: models } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id } },
    });

    const jobs = useProjectTrainingJobs(project_id);
    const [retrainModel, setRetrainModel] = useState<SchemaModel | null>(null);
    const [logsSourceId, setLogsSourceId] = useState<string | undefined>();

    const handleViewLogs = (model: SchemaModel) => {
        if (!model.train_job_id) {
            return;
        }

        setLogsSourceId(model.train_job_id);
    };

    const { addJob } = useJobUpdates(project_id);

    const hasModels = models.length > 0;
    const hasJobs = jobs.length > 0;
    const showIllustratedMessage = !hasModels && !hasJobs;

    return (
        <View height='100%' padding={'size-300'} UNSAFE_style={{ overflowY: 'auto' }}>
            <Flex direction={'column'} height={'100%'}>
                {showIllustratedMessage ? (
                    <NoModelsPlaceholder />
                ) : (
                    <Flex direction={'column'} flex={1} gap={'size-300'}>
                        <Flex justifyContent={'end'}>
                            <DialogTrigger>
                                <Button variant='accent'>Train model</Button>
                                {(close) => (
                                    <TrainModelDialog
                                        close={(job) => {
                                            if (job) addJob(job);
                                            close();
                                        }}
                                    />
                                )}
                            </DialogTrigger>
                        </Flex>
                        <JobList
                            jobs={jobs}
                            onViewLogs={(job) => {
                                setLogsSourceId(job.id);
                            }}
                        />
                        {hasModels && (
                            <ModelList
                                models={models}
                                jobs={jobs}
                                onRetrain={setRetrainModel}
                                onViewLogs={handleViewLogs}
                            />
                        )}
                    </Flex>
                )}
            </Flex>
            <DialogContainer onDismiss={() => setRetrainModel(null)}>
                {retrainModel && (
                    <TrainModelDialog
                        baseModel={retrainModel}
                        close={(job) => {
                            if (job) addJob(job);
                            setRetrainModel(null);
                        }}
                    />
                )}
            </DialogContainer>
            <DialogContainer type='fullscreen' onDismiss={() => setLogsSourceId(undefined)}>
                {logsSourceId != null && (
                    <LogsDialog close={() => setLogsSourceId(undefined)} initialSourceId={`job-${logsSourceId}`} />
                )}
            </DialogContainer>
        </View>
    );
};
