import { Suspense } from 'react';

import { Content, Grid, Heading, IllustratedMessage, Loading, View } from '@geti/ui';
import { Outlet, redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ReactComponent as RobotIllustration } from './assets/illustrations/INTEL_08_NO-TESTS.svg';
import { ErrorPage } from './components/error-page/error-page';
import { Camera } from './routes/cameras/camera';
import { Edit as CameraEdit } from './routes/cameras/edit';
import { Layout as CamerasLayout } from './routes/cameras/layout';
import { New as CamerasNew } from './routes/cameras/new';
import { CameraWebcam } from './routes/cameras/webcam';
import { Index as Datasets } from './routes/datasets/index';
import { Index as RecordingPage } from './routes/datasets/record/index';
import { Edit as EnvironmentEdit } from './routes/environments/edit';
import { Layout as EnvironmentsLayout } from './routes/environments/layout';
import { New as EnvironmentNew } from './routes/environments/new';
import { EnvironmentShow } from './routes/environments/show';
import { Index as Models } from './routes/models/index';
import { Index as Inference } from './routes/models/inference/index';
import { OpenApi } from './routes/openapi';
import { Index as Projects } from './routes/projects/index';
import { ProjectLayout } from './routes/projects/project.layout';
import { Edit as RobotEdit } from './routes/robots/edit';
import { Layout as RobotsLayout } from './routes/robots/layout';
import { New as RobotsNew } from './routes/robots/new';
import { NewRobotLayout } from './routes/robots/new-layout';
import { Robot } from './routes/robots/robot';
import { SO101Setup } from './routes/robots/so101-setup';
import { TabNavigation as RobotsTabNavigation } from './routes/robots/tab-navigation';

const root = path('/');
const projects = root.path('/projects');
const project = root.path('/projects/:project_id');
const robots = project.path('robots');
const robot = robots.path(':robot_id');
const datasets = project.path('/datasets');
const dataset = datasets.path(':dataset_id');
const models = project.path('/models');
const cameras = project.path('cameras');
const environments = project.path('environments');
const environment = environments.path(':environment_id');

export const paths = {
    root,
    openapi: root.path('/openapi'),
    projects: {
        index: projects,
    },
    project: {
        index: project,
        datasets: {
            index: datasets,
            show: dataset,
            record: dataset.path('record'),
        },
        robots: {
            index: robots,
            new: robots.path('new'),
            so101Setup: robots.path('new/so101-setup'),
            edit: robot.path('edit'),
            show: robot,
        },
        cameras: {
            index: cameras,
            webcam: cameras.path('/webcam'),
            new: cameras.path('/new'),
            edit: cameras.path(':camera_id/edit'),
            show: cameras.path(':camera_id'),
        },
        environments: {
            index: environments,
            new: environments.path('/new'),
            edit: environments.path(':environment_id/edit'),
            show: environments.path(':environment_id'),
            overview: environment,
            datasets: environment.path('/datasets'),
            models: environment.path('/models'),
        },
        models: {
            index: models,
            inference: models.path('/:model_id/inference'),
        },
    },
};

const EmptySelection = ({ heading, content }: { heading: string; content: string }) => {
    return (
        <IllustratedMessage>
            <RobotIllustration />
            <Content>{content}</Content>
            <Heading>{heading}</Heading>
        </IllustratedMessage>
    );
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <Outlet />
            </Suspense>
        ),
        errorElement: <ErrorPage />,
        children: [
            {
                index: true,
                loader: () => {
                    return redirect(paths.projects.index({}));
                },
            },
            {
                path: paths.projects.index.pattern,
                children: [
                    {
                        index: true,
                        element: <Projects />,
                    },
                ],
            },
            {
                path: paths.project.datasets.record.pattern,
                element: <RecordingPage />,
            },
            {
                path: paths.project.index.pattern,
                element: <ProjectLayout />,
                children: [
                    {
                        index: true,
                        loader: ({ params }) => {
                            if (params.project_id === undefined) {
                                return redirect(paths.projects.index({}));
                            }

                            return redirect(
                                paths.project.robots.index({
                                    project_id: params.project_id,
                                })
                            );
                        },
                    },
                    {
                        path: paths.project.datasets.index.pattern,
                        children: [
                            {
                                index: true,
                                element: <Datasets />,
                            },
                            {
                                path: paths.project.datasets.show.pattern,
                                element: <Datasets />,
                            },
                        ],
                    },
                    {
                        path: paths.project.models.index.pattern,
                        children: [
                            {
                                index: true,
                                element: <Models />,
                            },
                            {
                                path: paths.project.models.inference.pattern,
                                element: <Inference />,
                            },
                        ],
                    },
                    {
                        // robots
                        element: (
                            <Grid
                                areas={['header', 'content']}
                                UNSAFE_style={{
                                    gridTemplateRows: 'min-content auto',
                                }}
                                minHeight={0}
                                height={'100%'}
                            >
                                <RobotsTabNavigation />
                                <View gridArea='content'>
                                    <Outlet />
                                </View>
                            </Grid>
                        ),
                        children: [
                            // Robots
                            {
                                path: paths.project.robots.new.pattern,
                                element: <NewRobotLayout />,
                                children: [
                                    {
                                        index: true,
                                        element: <RobotsNew />,
                                    },
                                    {
                                        path: 'so101-setup',
                                        element: <SO101Setup />,
                                    },
                                ],
                            },
                            {
                                path: paths.project.robots.edit.pattern,
                                element: <RobotEdit />,
                            },
                            {
                                path: paths.project.robots.index.pattern,
                                element: <RobotsLayout />,
                                children: [
                                    {
                                        index: true,
                                        element: (
                                            <EmptySelection
                                                heading='Robots'
                                                content='Select a robot or create a new one'
                                            />
                                        ),
                                    },
                                    {
                                        path: paths.project.robots.show.pattern,
                                        element: <Robot />,
                                    },
                                ],
                            },
                            // Cameras
                            {
                                path: paths.project.cameras.new.pattern,
                                element: <CamerasNew />,
                            },
                            {
                                path: paths.project.cameras.edit.pattern,
                                element: <CameraEdit />,
                            },
                            {
                                path: paths.project.cameras.index.pattern,
                                element: <CamerasLayout />,
                                children: [
                                    {
                                        index: true,
                                        element: (
                                            <EmptySelection
                                                heading='Cameras'
                                                content='Select a camera or create a new one'
                                            />
                                        ),
                                    },
                                    {
                                        path: paths.project.cameras.show.pattern,
                                        element: <Camera />,
                                    },
                                    {
                                        path: paths.project.cameras.webcam.pattern,
                                        element: <CameraWebcam />,
                                    },
                                ],
                            },
                            // Environments
                            {
                                path: paths.project.environments.new.pattern,
                                element: <EnvironmentNew />,
                            },
                            {
                                path: paths.project.environments.edit.pattern,
                                element: <EnvironmentEdit />,
                            },
                            {
                                path: paths.project.environments.index.pattern,
                                element: <EnvironmentsLayout />,
                                children: [
                                    {
                                        index: true,
                                        element: (
                                            <EmptySelection
                                                heading='Environments'
                                                content='Select an environment or create a new one'
                                            />
                                        ),
                                    },
                                    {
                                        path: paths.project.environments.show.pattern,
                                        element: <EnvironmentShow />,
                                    },
                                    {
                                        path: paths.project.environments.datasets.pattern,
                                        element: <EnvironmentShow />,
                                    },
                                    {
                                        path: paths.project.environments.models.pattern,
                                        element: <EnvironmentShow />,
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                path: paths.openapi.pattern,
                element: <OpenApi />,
            },
            {
                path: '*',
                loader: () => {
                    return redirect(paths.projects.index({}));
                },
            },
        ],
    },
]);
