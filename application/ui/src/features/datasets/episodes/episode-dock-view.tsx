import { Suspense, useEffect, useRef } from 'react';

import { Flex, Loading, View } from '@geti/ui';
import {
    DockviewApi,
    DockviewReact,
    DockviewReadyEvent,
    IDockviewPanelProps,
    IDockviewReactProps,
} from 'dockview-react';

import { SchemaDatasetOutput, SchemaEpisode, SchemaEpisodeVideo } from '../../../api/openapi-spec';
import { EpisodeVideoCell } from './episode-video-cell.component';

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems={'center'} justifyContent={'center'}>
            <Loading mode='inline' />
        </Flex>
    );
};

const components = {
    robot: (props: IDockviewPanelProps<{ title: string; robot_id: string }>) => {
        return <div style={{ padding: '20px', color: 'white' }}>{props.params.title}</div>;
    },
    camera: (props: IDockviewPanelProps<{ title: string; video: SchemaEpisodeVideo, datasetId: string }>) => {
        return <EpisodeVideoCell episodeVideo={props.params.video} datasetId={props.params.datasetId}/>
    },
    default: (props: IDockviewPanelProps<{ title: string }>) => {
        return <div style={{ padding: '20px', color: 'white' }}>{props.params.title}</div>;
    },
} satisfies IDockviewReactProps['components'];

const buildDockviewPanels = (api: DockviewReadyEvent['api'], episode: SchemaEpisode, dataset: SchemaDatasetOutput) => {
    if (episode === null) {
        return api;
    }

    const panels = new Set<string>();

    Object.keys(episode.videos).forEach((videoId) => {
        const cameraName = videoId.replace('observation.images.', '')
        panels.add(cameraName);
        if (!api.panels.some((panel) => panel.id === cameraName)) {
            api.addPanel({
                id: cameraName,
                title: cameraName,
                component: 'camera',
                params: {
                    title: cameraName,
                    video: episode.videos[videoId],
                    datasetId: dataset.id
                },
                position: {
                    direction: 'left',
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
}

export const EpisodeDockView = ({episode, dataset}: EpisodeViewerProps) => {
    const api = useRef<DockviewApi>(null);

    const onReady = (event: DockviewReadyEvent): void => {
      api.current = buildDockviewPanels(event.api, episode, dataset);
    };

    useEffect(() => {
        if (!api.current) {
            return;
        }

        buildDockviewPanels(api.current, episode, dataset);
    }, [episode, dataset]);

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
