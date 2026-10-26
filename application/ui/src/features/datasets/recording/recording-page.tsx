import { Flex, Grid, minmax, ToastQueue, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { AppFooter } from '../../../components/app-footer/app-footer';
import { CollapsibleSidebarLayout } from '../../../components/collapsible-sidebar-layout/collapsible-sidebar-layout';
import { RobotControlView } from '../../robots/robot-control/robot-control-view';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { RuntimeSessionProvider, useRuntimeSession } from '../../robots/runtime-session-provider';
import { useDatasetId } from '../use-dataset';
import { Initializing } from './initializing';
import { RecordingProvider } from './recording-provider';
import { RecordingSidebar } from './recording-sidebar';
import { RecordingTopBar } from './recording-top-bar';

const RecordingContent = () => {
    const { state, readyForRecording, environment, observation, actions } = useRuntimeSession();

    if (!readyForRecording) {
        return <Initializing />;
    }

    return (
        <CollapsibleSidebarLayout side='right' width='size-5000' sidebar={<RecordingSidebar />}>
            <View padding='size-200' height='100%'>
                <RobotModelsProvider>
                    <Flex direction='column' height='100%' position='relative'>
                        <RobotControlView
                            environment={environment}
                            isReady={state.connected}
                            joints={{
                                get current() {
                                    return actions.current ?? observation.current;
                                },
                            }}
                        />
                    </Flex>
                </RobotModelsProvider>
            </View>
        </CollapsibleSidebarLayout>
    );
};

export const RecordingPage = () => {
    const { project_id, dataset_id } = useDatasetId();

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: { path: { dataset_id } },
    });

    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        { params: { path: { environment_id: dataset.environment_id, project_id } } }
    );

    return (
        <RuntimeSessionProvider environment={environment} dataset={dataset} onError={ToastQueue.negative}>
            <RecordingProvider>
                <Grid
                    areas={['header', 'content', 'footer']}
                    rows={['size-800', minmax(0, '1fr'), 'size-400']}
                    minHeight={0}
                    height='100%'
                >
                    <RecordingTopBar />
                    <View gridArea='content' minHeight={0} height='100%'>
                        <RecordingContent />
                    </View>
                    <AppFooter />
                </Grid>
            </RecordingProvider>
        </RuntimeSessionProvider>
    );
};
