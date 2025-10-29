import { $api, API_BASE_URL } from '../../api/client';
import { View, Well, Heading, Text, Content, Flex, Button, DialogTrigger, TableView, TableHeader, Column, TableBody, Cell, Row, Divider, Tag, ProgressBar, IllustratedMessage } from "@geti/ui"
import { SchemaTrainJob, TrainModelModal } from './train-model';
import useWebSocket from 'react-use-websocket';
import { SchemaJob, SchemaModel } from '../../api/openapi-spec';
import { useState } from 'react';

import { v4 as uuidv4 } from 'uuid'
import { useQueryClient } from '@tanstack/react-query';
import { useProjectId } from '../../features/projects/use-project';

import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';


const ModelList = ({ models }: { models: SchemaModel[] }) => {
    const sortedModels = models.toSorted((a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime())

    return (
        <View borderTopWidth='thin' borderTopColor='gray-400' backgroundColor={'gray-300'}>
            <TableView
                aria-label='Models'
                overflowMode='wrap'
                selectionStyle='highlight'
                selectionMode='single'
            >
                <TableHeader>
                    <Column>MODEL NAME</Column>
                    <Column>TRAINED</Column>
                    <Column>ARCHITECTURE</Column>
                    <Column>{''}</Column>
                </TableHeader>
                <TableBody>
                    {sortedModels.map((model) => (
                        <Row key={model.id}>
                            <Cell>{model.name}</Cell>
                            <Cell>{new Date(model.created_at!).toLocaleString()}</Cell>
                            <Cell>{model.policy}</Cell>
                            <Cell>Run model</Cell>
                        </Row>
                    ))}
                </TableBody>
            </TableView>
        </View>
    )
}


const ModelInTraining = ({trainJob}: {trainJob: SchemaTrainJob}) => {
    const cancelMutation = $api.useMutation('post', '/api/jobs/{job_id}/interrupt')
    const onCancel = () => {
        if (trainJob?.id !== undefined) {
            cancelMutation.mutate({
                params: {
                    path: {
                        job_id: trainJob.id!,
                    }
                },
            })
        }
    }


    if (trainJob === undefined) {
        return <></>
    }

    return (
        <View marginBottom={'size-600'}>
            <Heading level={4} marginBottom={'size-100'}>Current Training</Heading>
            <View borderTopWidth='thin' borderTopColor='gray-400' backgroundColor={'gray-300'}>
                <TableView
                    aria-label='Models'
                    overflowMode='wrap'
                    selectionStyle='highlight'
                    selectionMode='single'
                >
                    <TableHeader>
                        <Column>MODEL NAME</Column>
                        <Column>TRAINED</Column>
                        <Column>ARCHITECTURE</Column>
                        <Column>{''}</Column>
                    </TableHeader>
                    <TableBody>
                        <Row key={trainJob.id ?? uuidv4()}>
                            <Cell>{trainJob.payload.model_name}</Cell>
                            <Cell>...</Cell>
                            <Cell>{trainJob.payload.policy}</Cell>
                            <Cell>
                                <Button variant='secondary' onPress={onCancel}>Cancel</Button>
                            </Cell>
                        </Row>
                    </TableBody>
                </TableView>
            </View>
            {trainJob.status === "running" && <ProgressBar width={'100%'} value={trainJob.progress} />}
        </View>
    )
}


export const Index = () => {
    const { project_id } = useProjectId()
    const { data: models } = $api.useQuery('get', '/api/models/{project_id}', { params: { path: { project_id } } })

    const {} = useWebSocket(`${API_BASE_URL}/api/jobs/ws`, {
        onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
    });
    const client = useQueryClient();

    const [trainJob, setTrainJob] = useState<SchemaTrainJob>();

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as { event: string, data: SchemaJob };
        if (message.event === 'JOB_UPDATE') {
            if (message.data.status === "completed") {
                client.invalidateQueries({ queryKey: ['get', '/api/models/{project_id}'] });

                setTrainJob(undefined)
            } else {
                setTrainJob(message.data as SchemaTrainJob)
            }
        }
    };

    const hasModels = (models ?? []).length > 0
    const showIllustratedMessage = !hasModels && !trainJob

    return (
        <Flex height="100%">
            <Flex margin={'size-200'} direction={'column'} flex>
                <Heading level={4}>Models</Heading>
                <Divider size='S' marginTop='size-100' marginBottom={'size-100'} />
                {showIllustratedMessage
                    ? <Well flex UNSAFE_style={{ backgroundColor: "rgb(60,62,66)" }}>
                        <IllustratedMessage >
                            <EmptyIllustration />
                            <Content> Currently there are no trained models available. </Content>
                            <Text>If you've recorded a dataset it's time to begin training your model. </Text>
                            <Heading>No trained models</Heading>
                            <View margin={'size-100'}>
                                <DialogTrigger>
                                    <Button variant='accent'>Train model</Button>
                                    {TrainModelModal}
                                </DialogTrigger>
                            </View>
                        </IllustratedMessage>
                    </Well>
                    : <View margin={'size-300'}>
                        <Flex justifyContent={'end'} marginBottom='size-300'>
                            <DialogTrigger >
                                <Button variant='secondary'>Train model</Button>
                                {(close) => TrainModelModal((job) => {setTrainJob(job); close()})}
                            </DialogTrigger>
                        </Flex>
                        {trainJob && <ModelInTraining trainJob={trainJob} />}
                        {hasModels && <ModelList models={models ?? []} />}
                    </View>
                }
            </Flex>
        </Flex>
    )
};
