import { $api, API_BASE_URL } from '../../api/client';
import { View, Well, Heading, Text, Flex, Button, DialogTrigger, TableView, TableHeader, Column, TableBody, Cell, Row, Divider, Tag, ProgressBar } from "@geti/ui"
import { TrainModelModal } from './train-model';
import useWebSocket from 'react-use-websocket';
import { SchemaJob, SchemaTrainJobPayload } from '../../api/openapi-spec';
import { useState } from 'react';

import { v4 as uuidv4 } from 'uuid'

type SchemaTrainJob = Omit<SchemaJob, 'payload'> & {
    payload: SchemaTrainJobPayload;
};

const ModelList = () => {
    const { data: models } = $api.useQuery('get', '/api/models')

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
                    {(models ?? []).map((model) => (
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


const ModelInTraining = () => {
    const {} = useWebSocket(`${API_BASE_URL}/api/jobs/ws`, {
        onMessage: (event: WebSocketEventMap['message']) => onMessage(event),
    });

    const [trainJob, setTrainJob] = useState<SchemaTrainJob>();

    const cancelMutation = $api.useMutation('post', '/api/jobs/{job_id}/interrupt')

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message = JSON.parse(data) as { event: string, data: SchemaJob };
        if (message.event === 'JOB_UPDATE') {
            console.log(message);
            setTrainJob(message.data as SchemaTrainJob)
        }
    };

    const onCancel = () => {
        if (trainJob?.id !== undefined) {
            cancelMutation.mutate({
                params: {
                    path: {
                        job_id: trainJob?.id!,
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
            <ProgressBar width={'100%'} value={trainJob.progress} />
        </View>
    )
}


export const Index = () => {
    return (
        <View margin={'size-200'}>
            <Heading level={4}>Models</Heading>
            <Divider size='S' />
            <View margin={'size-300'}>
                <Flex justifyContent={'end'}>
                    <DialogTrigger >
                        <Button variant='secondary'>Train model</Button>
                        {TrainModelModal}
                    </DialogTrigger>
                </Flex>
                <ModelInTraining />
                <ModelList />
            </View >
        </View>
    )
};
