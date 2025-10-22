import { Button, Flex, Heading, Item, ListView, ProgressCircle, Well } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';

import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { CameraView } from './camera-view';
import { useRecording } from './use-recording';

interface RecordingProps {
    setup: SchemaTeleoperationConfig;
}
export const Recording = ({ setup }: RecordingProps) => {
    const { project_id } = useProjectId();
    const { startRecording, saveEpisode, cancelEpisode, observation, state, numberOfRecordings } = useRecording(setup);
    const client = useQueryClient();
    const navigate = useNavigate();

    const onDone = () => {
        client.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/datasets/{repo}/{id}'] });
        navigate(paths.project.datasets.index({ project_id }));
    };

    if (state.initialized) {
        return (
            <Well height={'100%'}>
                <Flex flex height={'100%'} gap='size-150'>
                    <Flex flex direction='column' maxWidth='size-2000'>
                        <ListView
                            minWidth={'size-3000'}
                            selectedKeys={new Set([0])}
                            selectionMode='single'
                            selectionStyle='highlight'
                            flex={'1 0 0'}
                        >
                            {state.is_recording ? <Item key={0}>Recording...</Item> : <></>}
                            <>
                                {[...Array(numberOfRecordings)].map((_, i) => (
                                    <Item key={numberOfRecordings - i}>{`Episode ${numberOfRecordings - i}`}</Item>
                                ))}
                            </>
                        </ListView>
                    </Flex>

                    <Flex direction={'column'} flex={0} gap='size-100' justifyContent={'start'}>
                        {setup.cameras.map((camera) => (
                            <CameraView key={camera.id} camera={camera} observation={observation} />
                        ))}
                    </Flex>

                    <Flex direction={'column'} justifyContent={'space-between'} alignItems={'end'} flex={1}>
                        {state.is_recording ? (
                            <Flex direction={'row'} gap='size-100'>
                                <Button isDisabled={saveEpisode.isPending} variant={'negative'} onPress={cancelEpisode}>
                                    Cancel
                                </Button>
                                <Button isPending={saveEpisode.isPending} onPress={() => saveEpisode.mutate()}>
                                    Save
                                </Button>
                            </Flex>
                        ) : (
                            <Button onPress={startRecording}>Start Episode</Button>
                        )}
                        <Button onPress={onDone}>Done</Button>
                    </Flex>
                </Flex>
            </Well>
        );
    } else {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'}>
                <Heading>Initializing</Heading>
                <ProgressCircle isIndeterminate />
            </Flex>
        );
    }
};
