import { $api, API_BASE_URL } from '../../api/client';
import { View, Heading, Text, Flex, Button, DialogTrigger, TableView, TableHeader, Column, TableBody, Cell, Row, Divider } from "@geti/ui"
import { TrainModelModal } from './train-model';
import useWebSocket from 'react-use-websocket';

const JobTestWidget = () => {
    const { lastJsonMessage } = useWebSocket(`${API_BASE_URL}/api/jobs/ws`);
    console.log(lastJsonMessage)
    return (
        <Text>{JSON.stringify(lastJsonMessage) ?? ""}</Text>
    )
}

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

export const Index = () => {
    return (
        <View margin={'size-200'}>
            <Heading level={4}>Models</Heading>
            <Divider size='S' />
            <View margin={'size-300'}>
                <Flex justifyContent={'end'}>
                    <DialogTrigger>
                        <Button variant='secondary'>Train model</Button>
                        {TrainModelModal}
                    </DialogTrigger>
                </Flex>
                <JobTestWidget/>
                <Text>Active training model goes here..</Text>
                <ModelList />
            </View >
        </View>
    )
};
