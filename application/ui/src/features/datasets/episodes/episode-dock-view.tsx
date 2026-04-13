import { Suspense, useEffect, useRef } from 'react';

import { Flex, Loading, View } from '@geti-ui/ui';
import {
    DockviewApi,
    DockviewReact,
    DockviewReadyEvent,
    IDockviewPanelProps,
    IDockviewReactProps,
} from 'dockview-react';

import {
    SchemaDatasetOutput,
    SchemaEnvironmentWithRelations,
    SchemaEpisode,
    SchemaEpisodeVideo,
} from '../../../api/openapi-spec';
import { EpisodeVideoCell } from './episode-video-cell.component';
import { RobotCell } from './robot-cell.component';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

const components = {
    robot: (props: IDockviewPanelProps<{ robot_id: string }>) => {
        return <RobotCell robotId={props.params.robot_id} />;
    },
    camera: (props: IDockviewPanelProps<{ video: SchemaEpisodeVideo; datasetId: string }>) => {
        return <EpisodeVideoCell episodeVideo={props.params.video} datasetId={props.params.datasetId} />;
    },
    default: (props: IDockviewPanelProps<{ title: string }>) => {
        return <div style={{ padding: '20px', color: 'white' }}>{props.params.title}</div>;
    },
} satisfies IDockviewReactProps['components'];

const buildDockviewPanels = (
    api: DockviewReadyEvent['api'],
    episode: SchemaEpisode,
    dataset: SchemaDatasetOutput,
    environment: SchemaEnvironmentWithRelations
) => {
    if (episode === null) {
        return api;
    }

    const panels = new Set<string>();

    Object.keys(episode.videos).forEach((videoId) => {
        const cameraName = videoId.replace('observation.images.', '');
        panels.add(cameraName);
        if (!api.panels.some((panel) => panel.id === cameraName)) {
            api.addPanel({
                id: cameraName,
                title: cameraName,
                component: 'camera',
                params: {
                    video: episode.videos[videoId],
                    datasetId: dataset.id,
                },
                position: {
                    direction: 'left',
                    referencePanel: '',
                },
            });
        }
    });

    environment.robots?.forEach((robot) => {
        panels.add(robot.robot.id);
        if (!api.panels.some((panel) => panel.id === robot.robot.id)) {
            api.addPanel({
                id: robot.robot.id,
                params: { robot_id: robot.robot.id },
                title: robot.robot.name,
                component: 'robot',

                position: {
                    direction: 'below',
                    referencePanel: '',
                },
            });
        }
    });
    // Remove any panels that are no longer part of the environment
    api.panels
        .filter((panel) => panels.has(panel.id) === false)
        .forEach((panel) => {
            api.removePanel(panel);
        });

    return api;
};

interface EpisodeViewerProps {
    episode: SchemaEpisode;
    dataset: SchemaDatasetOutput;
    environment: SchemaEnvironmentWithRelations;
}

export const EpisodeDockView = ({ episode, dataset, environment }: EpisodeViewerProps) => {
    const api = useRef<DockviewApi>(null);

    const onReady = (event: DockviewReadyEvent): void => {
        api.current = buildDockviewPanels(event.api, episode, dataset, environment);
    };

    useEffect(() => {
        if (!api.current) {
            return;
        }

        buildDockviewPanels(api.current, episode, dataset, environment);
    }, [episode, dataset, environment]);

    return (
        <View flex>
            <View backgroundColor={'gray-200'} height={'100%'} maxHeight='100vh' position={'relative'}>
                <Suspense fallback={<CenteredLoading />}>
                    <DockviewReact onReady={onReady} components={components} />
                </Suspense>
            </View>
        </View>
    );
};
