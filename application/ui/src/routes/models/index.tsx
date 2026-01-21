import {
    Button,
    Content,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    IllustratedMessage,
    Text,
    View,
    Well,
} from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import useWebSocket from 'react-use-websocket';
import { $api } from '../../api/client';
import { SchemaJob, SchemaModel } from '../../api/openapi-spec';
import { useProjectId } from '../../features/projects/use-project';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { SchemaTrainJob, TrainModelModal } from './train-model';

import { ModelHeader, ModelRow } from './model-table.component';
import { TrainingHeader, TrainingRow } from './job-table.component';

const ModelList = ({ models }: { models: SchemaModel[] }) => {
    const sortedModels = models.toSorted(
        (a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime()
    );

    const deleteModelMutation = $api.useMutation('delete', '/api/models');

    const deleteModel = (model: SchemaModel) => {
        deleteModelMutation.mutate({ params: { query: { uuid: model.id! } } });
    };

    return (
        <View marginBottom={'size-600'}>
            <ModelHeader />
            {sortedModels.map((model) => (
                <ModelRow key={model.id} model={model} onDelete={() => deleteModel(model)} />
            ))}
        </View>
    );
};

const JobList = ({ jobs }: { jobs: SchemaTrainJob[] }) => {
    const sortedJobs = jobs.filter((m) => m.status !== "completed").toSorted(
        (a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime()
    );

    const interruptMutation = $api.useMutation('post', '/api/jobs/{job_id}:interrupt');
    const onInterrupt = (job: SchemaTrainJob) => {
        if (job.id !== undefined) {
            interruptMutation.mutate({
                params: {
                    query: {
                        uuid: job.id,
                    },
                },
            });
        }
    };

    return (
        <View marginBottom={'size-600'}>
            <Heading level={4} marginBottom={'size-100'}>
                Current Training
            </Heading>

            <TrainingHeader />
            { sortedJobs.map((job) => <TrainingRow trainJob={job} onInterrupt={() => onInterrupt(job)} />) }
        </View>
    );
};

const useProjectJobs = (project_id: string): SchemaJob[] => {
    const {data: allJobs } = $api.useQuery('get', '/api/jobs')

    return allJobs?.filter((j) => j.project_id === project_id) ?? []
}

export const Index = () => {
    const { project_id } = useProjectId();
    const { data: models } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id } },
    });

    const jobs = useProjectJobs(project_id);

    const {} = useWebSocket(`/api/jobs/ws`, {
        shouldReconnect: () => true,
        onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
    });
    const client = useQueryClient();

    const updateJob = (job: SchemaJob) => {
        client.setQueryData<SchemaJob[]>(["get", "/api/jobs"], (old = []) => {
            return old.map((m) => m.id === job.id ? job : m);
        })
    }

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message_data = JSON.parse(data);
        if (message_data.event === 'JOB_UPDATE') {
            const message = message_data as { event: string; data: SchemaJob };
            if (message.data.project_id !== project_id) {
                return;
            }

            updateJob(message.data as SchemaTrainJob)
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
                                    {TrainModelModal}
                                </DialogTrigger>
                            </View>
                        </IllustratedMessage>
                    </Well>
                ) : (
                    <View margin={'size-300'}>
                        <Flex justifyContent={'end'} marginBottom='size-300'>
                            <DialogTrigger>
                                <Button variant='secondary'>Train model</Button>
                                {(close) =>
                                    TrainModelModal((job) => {
                                        //setTrainJob(job);
                                        close();
                                    })
                                }
                            </DialogTrigger>
                        </Flex>
                        <JobList jobs={jobs.filter((m) => m.type === "training") as SchemaTrainJob[]} />
                        {hasModels && <ModelList models={models} />}
                    </View>
                )}
            </Flex>
        </Flex>
    );
};
