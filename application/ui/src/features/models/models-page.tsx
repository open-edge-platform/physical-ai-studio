import { useState } from 'react';

import { Button, DialogContainer, DialogTrigger, Divider, Flex, View } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { SchemaModel } from '../../api/openapi-spec';
import { useJobCache } from '../jobs/use-job-cache';
import { LogsDialog } from '../logs/logs-dialog';
import { useProjectId } from '../projects/use-project';
import { JobList } from './job-table/job-list';
import { ModelsList } from './models-table/models-list';
import { NoModelsPlaceholder } from './no-models-placeholder';
import { SchemaTrainJob, TrainModelDialog } from './train-model-dialog/train-model-dialog';
import { useProjectTrainingJobs } from './use-project-training-jobs';

const isActiveJob = (job: SchemaTrainJob) => job.status === 'running' || job.status === 'pending';

export const ModelsPage = () => {
    const { project_id } = useProjectId();
    const { data: models } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id } },
    });

    const jobs = useProjectTrainingJobs(project_id);
    const activeJobs = jobs.filter(isActiveJob);

    const [retrainModel, setRetrainModel] = useState<SchemaModel | null>(null);
    const [logsSourceId, setLogsSourceId] = useState<string | undefined>();

    const { addJob } = useJobCache();

    const handleViewLogs = (model: SchemaModel) => {
        if (!model.train_job_id) {
            return;
        }

        setLogsSourceId(model.train_job_id);
    };

    const hasModels = models.length > 0;

    const showIllustratedMessage = !hasModels && activeJobs.length === 0;

    return (
        <View height='100%' padding={'size-300'} UNSAFE_style={{ overflowY: 'auto' }}>
            <Flex direction={'column'} height={'100%'}>
                {showIllustratedMessage ? (
                    <NoModelsPlaceholder onJobCreated={addJob} />
                ) : (
                    <Flex direction={'column'} flex={1} gap={'size-300'} minHeight={0}>
                        <Flex justifyContent={'end'} alignItems={'center'}>
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
                        <Divider size={'S'} />
                        <Flex
                            direction={'column'}
                            flex={1}
                            minHeight={0}
                            gap={'size-600'}
                            UNSAFE_style={{
                                overflowY: 'auto',
                                scrollbarGutter: 'stable',
                                paddingLeft: 'var(--spectrum-global-dimension-size-350)',
                            }}
                        >
                            <JobList
                                jobs={activeJobs}
                                onViewLogs={(job) => {
                                    setLogsSourceId(job.id);
                                }}
                            />
                            {hasModels && (
                                <ModelsList
                                    models={models}
                                    jobs={jobs}
                                    onRetrain={setRetrainModel}
                                    onViewLogs={handleViewLogs}
                                />
                            )}
                        </Flex>
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
