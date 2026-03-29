import { Suspense } from 'react';

import { Flex, Grid, Icon, Link, Loading, Text, ToastQueue, View } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { useDatasetId } from '../../../features/datasets/use-dataset';
import { RobotControlProvider } from '../../../features/robots/robot-control-provider';
import { paths } from '../../../router';
import { RecordingViewer } from './recording-viewer';

import classes from './index.module.scss';

const RecordingPage = () => {
    const { project_id, dataset_id } = useDatasetId();

    const { data: dataset } = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}', {
        params: {
            path: {
                dataset_id,
            },
        },
    });

    const { data: environment } = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: {
                path: {
                    environment_id: dataset.environment_id,
                    project_id,
                },
            },
        }
    );

    return (
        <Grid
            areas={['header', 'content']}
            UNSAFE_style={{
                gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
            }}
            minHeight={0}
            height={'100%'}
        >
            <View backgroundColor={'gray-300'} gridArea={'header'}>
                <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                    <Link
                        href={paths.project.datasets.show({ project_id, dataset_id })}
                        isQuiet
                        variant='overBackground'
                    >
                        <Flex marginEnd='size-200' direction='row' gap='size-200' alignItems={'center'}>
                            <Icon>
                                <ChevronLeft />
                            </Icon>
                            <Flex direction={'column'}>
                                <Text UNSAFE_className={classes.headerText}>Adding Episode</Text>
                                <Text UNSAFE_className={classes.subHeaderText}>
                                    {environment.name} | {dataset.default_task}
                                </Text>
                            </Flex>
                        </Flex>
                    </Link>
                </Flex>
            </View>

            <View gridArea={'content'} maxHeight={'100vh'} minHeight={0} height='100%'>
                <View padding='size-200' height='100%'>
                    <RobotControlProvider environment={environment} dataset={dataset} onError={ToastQueue.negative}>
                        <RecordingViewer />
                    </RobotControlProvider>
                </View>
            </View>
        </Grid>
    );
};

export const Index = () => {
    return (
        <Suspense fallback={<Loading mode='overlay' />}>
            <RecordingPage />
        </Suspense>
    );
};
