import { Suspense, useState } from 'react';

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
    minmax,
    ProgressCircle,
    Text,
    View,
} from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { SchemaDatasetOutput } from '../../../api/openapi-spec';
import { AppFooter } from '../../../components/app-footer/app-footer';
import { useDatasetId } from '../../../features/datasets/use-dataset';
import {
    environmentDraft,
    RuntimeSessionProvider,
    useRuntimeSession,
} from '../../../features/runtime-session/runtime-session-provider';
import { SessionWorkspace } from '../../../features/runtime-session/workspace/session-workspace';
import { paths } from '../../../paths';

import classes from './index.module.css';

const TotalRecordedEpisodes = () => {
    const { dataset_id: opened } = useDatasetId();
    const { state } = useRuntimeSession();
    // The loaded dataset, which the side bar can swap; the one the page was opened with until then.
    const dataset_id = state?.dataset?.dataset_id ?? opened;

    const episodeQuery = $api.useSuspenseQuery('get', '/api/dataset/{dataset_id}/episodes', {
        params: {
            path: {
                dataset_id,
            },
        },
    });
    // Episodes saved in this session land in the dataset only when the recording is finished.
    const totalEpisodes = Number(episodeQuery.data?.length) + (state?.dataset?.episodes_recorded ?? 0);

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

/**
 * Back to the dataset, after the recording is finished.
 *
 * Unloading the dataset copies what this session recorded into it, so waiting for that ack
 * before leaving means the dataset page shows the new episodes.
 */
const FinishRecordingLink = ({ subtitle }: { subtitle: string }) => {
    const { project_id, dataset_id } = useDatasetId();
    const { state, send } = useRuntimeSession();
    const queryClient = useQueryClient();
    const navigate = useNavigate();
    const [isFinishing, setIsFinishing] = useState(false);

    const finish = async () => {
        setIsFinishing(true);
        try {
            if (state?.dataset) {
                await send({ command: 'unload_dataset' });
            }
            await queryClient.invalidateQueries({ queryKey: ['get', '/api/dataset/{dataset_id}/episodes'] });
        } finally {
            setIsFinishing(false);
        }
        navigate(paths.project.datasets.show({ project_id, dataset_id: state?.dataset?.dataset_id ?? dataset_id }));
    };

    return (
        <Link onPress={() => void finish()} isQuiet variant='overBackground'>
            {/* Without an href, Link puts its press handlers on its child, and Flex drops them. */}
            <div>
                <Flex marginEnd='size-200' direction='row' gap='size-200' alignItems={'center'}>
                    {isFinishing ? (
                        <ProgressCircle size='S' isIndeterminate aria-label='Saving the recording' />
                    ) : (
                        <Icon>
                            <ChevronLeft />
                        </Icon>
                    )}
                    <Flex direction={'column'}>
                        <Text UNSAFE_className={classes.headerText}>
                            {isFinishing ? 'Saving recording…' : 'Adding Episode'}
                        </Text>
                        <Text UNSAFE_className={classes.subHeaderText}>{subtitle}</Text>
                    </Flex>
                </Flex>
            </div>
        </Link>
    );
};

const RecordingSession = ({ dataset }: { dataset: SchemaDatasetOutput }) => (
    <SessionWorkspace
        initialMode='record'
        cancelHref={paths.project.datasets.show({ project_id: dataset.project_id, dataset_id: dataset.id! })}
    />
);

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
        <RuntimeSessionProvider
            project_id={project_id}
            environment={environmentDraft(environment)}
            dataset_id={dataset_id}
        >
            <Grid
                areas={['header', 'content', 'footer']}
                UNSAFE_style={{
                    // Footer track matches ProjectLayout's so the strip is the
                    // same height here as on every other page.
                    // eslint-disable-next-line max-len
                    gridTemplateRows: `var(--spectrum-global-dimension-size-800, 4rem) ${minmax(0, '1fr')} var(--spectrum-global-dimension-size-400)`,
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
                        <FinishRecordingLink subtitle={`${environment.name} | ${dataset.default_task}`} />
                        <Suspense fallback={null}>
                            <TotalRecordedEpisodes />
                        </Suspense>
                    </Flex>
                </View>

                <View gridArea={'content'} maxHeight={'100vh'} minHeight={0} height='100%'>
                    <View padding='size-200' height='100%'>
                        {/* Inside the provider: anything that suspends while loading must not take the
                            page-level boundary, which would unmount the provider and end the session. */}
                        <Suspense fallback={<Loading mode='inline' />}>
                            <RecordingSession dataset={dataset} />
                        </Suspense>
                    </View>
                </View>
                <AppFooter />
            </Grid>
        </RuntimeSessionProvider>
    );
};

export const Index = () => {
    return (
        <Suspense fallback={<Loading mode='overlay' />}>
            <RecordingPage />
        </Suspense>
    );
};
