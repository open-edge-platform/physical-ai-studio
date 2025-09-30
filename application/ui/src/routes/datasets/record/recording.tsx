import { useCallback, useEffect } from 'react';

import { Button, Flex, Text, Heading, Item, ListView, ProgressCircle, Well } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';
import { ReadyState } from 'react-use-websocket';

import { $api } from '../../../api/client';
import { paths } from '../../../router';
import { CameraView } from './camera-view';
import { useRecording } from './use-recording';
import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { useProjectId } from '../../../features/projects/use-project';

interface RecordingProps {
    setup: SchemaTeleoperationConfig;
}
export const Recording = ({ setup }: RecordingProps) => {
    const { project_id } = useProjectId();
    const {
        init,
        startRecording,
        saveEpisode,
        cancelEpisode,
        cameraObservations,
        state,
        numberOfRecordings,
        disconnect,
        readyState,
    } = useRecording(setup);
    const client = useQueryClient();
    const navigate = useNavigate();

    //const saveProjectMutation = $api.useMutation('put', '/api/projects');

    const onDone = () => {
        client.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/datasets/{repo}/{id}'] });
        navigate(paths.project.datasets.index({ project_id }));
    };

    //const updateProject = useCallback(() => {
    //    saveProjectMutation
    //        .mutateAsync({
    //            body: setup.project,
    //        })
    //        .then(() => {
    //            client.invalidateQueries({ queryKey: ['get', '/api/projects/{id}'] });
    //        });
    //}, [client, setup.project, saveProjectMutation]);

    //useEffect(() => {
    //    if (readyState === ReadyState.OPEN && !state.initialized && init.isIdle) {
    //        init.mutate();
    //        //Overwrite current project to keep current camera config
    //        updateProject();
    //    }
    //}, [setup, readyState, updateProject, init, state.initialized]);


    const debug = false;
    if (debug) {
      return (
        <Flex>
          <Text>Readystate: {readyState}</Text>
          <Button onPress={() => init.mutate()}>initialize</Button>
          <Button onPress={startRecording}>record</Button>
          <Button onPress={cancelEpisode}>cancel</Button>
          <Button onPress={() => saveEpisode.mutate()}>save</Button>
          <Button onPress={disconnect}>disconnect</Button>
        </Flex>
      )

    }

    if (state.initialized) {
        return (
            <Well flex='1'>
                <Flex height={'100%'} gap='size-150'>
                    <ListView
                        minWidth={'size-3000'}
                        selectedKeys={new Set([0])}
                        selectionMode='single'
                        selectionStyle='highlight'
                    >
                        {state.is_recording ? <Item key={0}>Recording...</Item> : <></>}
                        <>
                            {[...Array(numberOfRecordings)].map((_, i) => (
                                <Item key={numberOfRecordings - i}>{`Episode ${numberOfRecordings - i}`}</Item>
                            ))}
                        </>
                    </ListView>

                    <Flex direction={'column'} flex={0} gap='size-100' justifyContent={'start'}>
                        {setup.cameras.map((camera) => (
                            <CameraView key={camera.id} camera={camera} cameraObservations={cameraObservations} />
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
            <Flex width='100%' alignItems={'center'} justifyContent={'center'}>
                <Heading>Initializing</Heading>
                <ProgressCircle isIndeterminate />
            </Flex>
        );
    }
};
