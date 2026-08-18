import { Button, Content, DialogTrigger, Flex, Heading, IllustratedMessage, Text, View } from '@geti-ui/ui';

import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { TrainModelDialog } from './train-model-dialog/train-model-dialog';

export const NoModelsPlaceholder = () => {
    return (
        <Flex margin={'size-200'} direction={'column'} height='100%'>
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
        </Flex>
    );
};
