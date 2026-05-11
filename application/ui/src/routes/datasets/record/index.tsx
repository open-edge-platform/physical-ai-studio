import { Suspense } from 'react';

import {
    Badge,
    Content,
    ContextualHelp,
    Flex,
    Footer,
    Grid,
    Heading,
    Icon,
    Link,
    Loading,
    Text,
    ToastQueue,
    View,
} from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { useDatasetId } from '../../../features/datasets/use-dataset';
import { RobotControlProvider, useRobotControl } from '../../../features/robots/robot-control-provider';
import { paths } from '../../../router';
import { RecordingViewer } from './recording-viewer';

import classes from './index.module.scss';

const TotalRecordedEpisodes = () => {
    const { dataset_id } = useDatasetId();
    const { state } = useRobotControl();

    const episodeQuery = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id,
            },
        },
    });
    const totalEpisodes = Number(episodeQuery.data?.length) + state.episodes_recorded;

    if (totalEpisodes === 0 || isNaN(totalEpisodes)) {
        return null;
    }

    return (
        <Flex direction='row' justifyContent={'space-between'} gap='size-150'>
            <Flex alignItems={'center'} gap='size-50'>
                <Text UNSAFE_className={classes.episodesText}>Total episodes recorded</Text>
                <ContextualHelp variant='info'>
                    <Heading>Recommended amount of episodes</Heading>
                    <Content>
                        <Text>We recommend recording at least 50 episodes before training your first model.</Text>
                    </Content>
                    <Footer>
                        <Link href='https://github.com/open-edge-platform/physical-ai-studio/issues/358'>
                            Learn more about recommended dataset sizes
                        </Link>
                    </Footer>
                </ContextualHelp>
            </Flex>
            <div className={classes.episodesCount}>
                <Badge variant='positive'>{totalEpisodes}</Badge>
            </div>
        </Flex>
    );
};

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
        <RobotControlProvider environment={environment} dataset={dataset} onError={ToastQueue.negative}>
            <Grid
                areas={['header', 'content']}
                UNSAFE_style={{
                    gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
                }}
                minHeight={0}
                height={'100%'}
            >
                <View backgroundColor={'gray-300'} gridArea={'header'}>
                    <Flex
                        height='100%'
                        alignItems={'center'}
                        marginX='1rem'
                        gap='size-200'
                        justifyContent={'space-between'}
                    >
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
                        <TotalRecordedEpisodes />
                    </Flex>
                </View>

                <View gridArea={'content'} maxHeight={'100vh'} minHeight={0} height='100%'>
                    <View padding='size-200' height='100%'>
                        <RecordingViewer />
                    </View>
                </View>
            </Grid>
        </RobotControlProvider>
    );
};

export const Index = () => {
    return (
        <Suspense fallback={<Loading mode='overlay' />}>
            <RecordingPage />
        </Suspense>
    );
};
