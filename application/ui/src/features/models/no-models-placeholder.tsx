import { Button, Content, DialogTrigger, Flex, Heading, IllustratedMessage, Text, View } from '@geti-ui/ui';

import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { SchemaTrainJob, TrainModelDialog } from './train-model-dialog/train-model-dialog';

export const NoModelsPlaceholder = ({ onJobCreated }: { onJobCreated?: (job: SchemaTrainJob) => void }) => {
    return (
        <Flex margin={'size-200'} direction={'column'} height='100%'>
            <IllustratedMessage>
                <EmptyIllustration />
                <Content> Currently there are no trained models available. </Content>
                <Text>If you&apos;ve recorded a dataset it&apos;s time to begin training your model. </Text>
                <Heading>No trained models</Heading>
                <View margin={'size-100'}>
                    <Flex gap={'size-100'} justifyContent={'center'}>
                        <DialogTrigger>
                            <Button variant='accent'>Train model</Button>
                            {(close) => (
                                <TrainModelDialog
                                    close={(job) => {
                                        if (job) onJobCreated?.(job);
                                        close();
                                    }}
                                />
                            )}
                        </DialogTrigger>
                    </Flex>
                </View>
            </IllustratedMessage>
        </Flex>
    );
};
