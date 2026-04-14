import { useState } from 'react';

import {
    Button,
    Content,
    DialogContainer,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    IllustratedMessage,
    Text,
    View,
    Well,
} from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import useWebSocket from 'react-use-websocket';

import { $api, fetchClient } from '../../api/client';
import { SchemaTrainJob as SchemaJob, SchemaModel } from '../../api/openapi-spec';
import { LogsDialog } from '../../features/logs/logs-dialog';
import { useProjectId } from '../../features/projects/use-project';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { TrainingHeader, TrainingRow } from './job-table.component';
import { ModelHeader, ModelRow } from './model-table.component';
import { SchemaTrainJob, TrainModelDialog } from './train-model-dialog';

const ModelList = ({
    models,
    jobs,
    onRetrain,
    onViewLogs,
}: {
    models: SchemaModel[];
    jobs: SchemaJob[];
    onRetrain: (model: SchemaModel) => void;
    onViewLogs: (model: SchemaModel) => void;
}) => {
    const sortedModels = models.toSorted(
        (a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime()
    );

    const jobsById = new Map(jobs.map((j) => [j.id, j]));

    const deleteModelMutation = $api.useMutation('delete', '/api/models/{model_id}');

    const deleteModel = (model: SchemaModel) => {
        deleteModelMutation.mutate({ params: { path: { model_id: model.id! } } });
    };

    return (
        <View marginBottom={'size-600'}>
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

const JobList = ({ jobs, onViewLogs }: { jobs: SchemaTrainJob[]; onViewLogs: (job: SchemaTrainJob) => void }) => {
    const sortedJobs = jobs
        .filter((m) => m.status !== 'completed')
        .toSorted((a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime());

    const interruptMutation = $api.useMutation('post', '/api/jobs/{job_id}:interrupt');
    const onInterrupt = (job: SchemaTrainJob) => {
        if (job.id !== undefined) {
            interruptMutation.mutate({
                params: {
                    path: {
                        job_id: job.id,
                    },
                },
            });
        }
    };

    if (sortedJobs.length === 0) {
        return <></>;
    }

    return (
        <View marginBottom={'size-600'}>
            <Heading level={4} marginBottom={'size-100'}>
                Current Training
            </Heading>

            <TrainingHeader />
            {sortedJobs.map((job) => (
                <TrainingRow
                    key={job.id}
                    trainJob={job}
                    onInterrupt={() => onInterrupt(job)}
                    onViewLogs={() => onViewLogs(job)}
                />
            ))}
        </View>
    );
};

const useProjectJobs = (project_id: string): SchemaJob[] => {
    const { data: allJobs } = $api.useQuery('get', '/api/jobs');

    return allJobs?.filter((j) => j.project_id === project_id) ?? [];
};

export const Index = () => {
    const { project_id } = useProjectId();
    const { data: models } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id } },
    });

    const jobs = useProjectJobs(project_id);
    const [retrainModel, setRetrainModel] = useState<SchemaModel | null>(null);
    const [logsSourceId, setLogsSourceId] = useState<string | undefined>();

    const handleViewLogs = (model: SchemaModel) => {
        if (!model.train_job_id) {
            return;
        }

        setLogsSourceId(model.train_job_id);
    };

    const {} = useWebSocket(fetchClient.PATH('/api/jobs/ws'), {
        shouldReconnect: () => true,
        onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
    });
    const client = useQueryClient();

    const updateJob = (job: SchemaJob) => {
        client.setQueryData<SchemaJob[]>(['get', '/api/jobs'], (old = []) => {
            return old.map((m) => (m.id === job.id ? job : m));
        });
    };

    const addJob = (job: SchemaJob) => {
        client.setQueryData<SchemaJob[]>(['get', '/api/jobs'], (old = []) => {
            return [...old, job];
        });
    };

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message_data = JSON.parse(data);
        if (message_data.event === 'JOB_UPDATE') {
            const message = message_data as { event: string; data: SchemaJob };
            if (message.data.project_id !== project_id) {
                return;
            }

            updateJob(message.data as SchemaTrainJob);
            if (message.data.status === 'completed') {
                client.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/models'] });
            }
        }
    };

    const hasModels = models.length > 0;
    const hasJobs = jobs.length > 0;
    const showIllustratedMessage = !hasModels && !hasJobs;

    return (
        <Flex height='100%'>
            <Flex margin={'size-200'} direction={'column'} flex>
                <Heading level={4}>Models</Heading>
                <Divider size='S' marginTop='size-100' marginBottom={'size-100'} />
                {showIllustratedMessage ? (
                    <Well flex UNSAFE_style={{ backgroundColor: 'rgb(60,62,66)' }}>
                        <IllustratedMessage>
                            <EmptyIllustration />
                            <Content> Currently there are no trained models available. </Content>
                            <Text>If you&apos;ve recorded a dataset it&apos;s time to begin training your model. </Text>
                            <Heading>No trained models</Heading>
                            <View margin={'size-100'}>
                                <DialogTrigger>
                                    <Button variant='accent'>Train model</Button>
                                    {(close) => <TrainModelDialog close={close} />}
                                </DialogTrigger>
                            </View>
                        </IllustratedMessage>
                    </Well>
                ) : (
                    <View margin={'size-300'}>
                        <Flex justifyContent={'end'} marginBottom='size-300'>
                            <DialogTrigger>
                                <Button variant='secondary'>Train model</Button>
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
                            jobs={jobs.filter((m) => m.type === 'training') as SchemaTrainJob[]}
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
                    </View>
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
        </Flex>
    );
};
