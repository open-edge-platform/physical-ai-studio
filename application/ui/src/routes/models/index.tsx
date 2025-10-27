import { $api } from '../../api/client';
import { View, Text, Flex, Button, DialogTrigger } from "@geti/ui"
import { TrainModelModal } from './train-model';

export const Index = () => {
    const {data: models } = $api.useQuery('get','/api/models')

    const showTrainModelModal = () => {
        console.log("...")
    }
    return (
        <View margin={'size-200'}>
            <Flex justifyContent={'space-between'}>
                <Text>Models</Text>
                <DialogTrigger>
                    <Button variant='secondary' onPress={showTrainModelModal}>Train model</Button>
                    {TrainModelModal}
                </DialogTrigger>
            </Flex>
            <Text>Active training model goes here..</Text>
            <Text>List goes here..</Text>
        </View>
    )
};
