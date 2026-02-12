import { Flex, View } from '@geti/ui';
import { useParams } from 'react-router';

import { $api } from '../../../api/client';
import { TeleoperationSetup } from '../../../features/configuration/teleoperation/teleoperation';
import { RecordingProvider, useRecording } from './recording-provider';
import { RecordingViewer } from './recording-viewer';

const RecordingPage = () => {
    const { recordingConfig, setRecordingConfig } = useRecording();
    const { dataset_id } = useParams();
    if (!dataset_id) {
        throw new Error('No dataset_id given.');
    }

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

    if (recordingConfig) {
        return (
            <View padding='size-200' height='100%'>
                <RecordingViewer recordingConfig={recordingConfig} />
            </View>
        );
    } else {
        return (
            <Flex margin={'size-200'} justifySelf='center' flex maxWidth={'size-6000'}>
                <TeleoperationSetup dataset={dataset} onDone={setRecordingConfig} />
            </Flex>
        );
    }
};

export const Index = () => {
    return (
        <RecordingProvider>
            <RecordingPage />
        </RecordingProvider>
    );
};
