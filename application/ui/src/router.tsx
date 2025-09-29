import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { Outlet, redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ErrorPage } from './components/error-page/error-page';
import { Camera, CameraOverview } from './routes/cameras/camera';
import { Layout as CamerasLayout } from './routes/cameras/layout';
import { CameraWebcam } from './routes/cameras/webcam';
import { Index as Datasets } from './routes/datasets/index';
import { Record } from './routes/datasets/record/record';
import { Index as Models } from './routes/models/index';
import { OpenApi } from './routes/openapi';
import { Index as Projects } from './routes/projects/index';
import { NewProjectPage } from './routes/projects/new';
import { ProjectLayout } from './routes/projects/project.layout';
import { Index as RobotConfiguration } from './routes/robot-configuration/index';

const root = path('/');
const projects = root.path('/projects');
const project = root.path('/projects/:project_id');
const robots = project.path('robots');
const robot = robots.path(':robot_id');
const datasets = project.path('/datasets');
const models = project.path('/models');
const cameras = project.path('cameras');

export const paths = {
    root,
    openapi: root.path('/openapi'),
    projects: {
        index: projects,
        new: projects.path('/new'),
    },
    project: {
        datasets: {
            index: datasets,
            record: datasets.path('/:dataset_id/record'),
            record_new: datasets.path('/record/new'),
        },
        cameras: {
            index: cameras,
            webcam: cameras.path('/webcam'),
            overview: cameras.path('/overview'),
            new: cameras.path('/new'),
            show: cameras.path(':camera_id'),
        },
        robotConfiguration: {
            index: robots,
        },
        models: {
            index: models,
        },
    },
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
                    {
                        path: paths.projects.new.pattern,
                        element: <NewProjectPage />,
                    },
                ],
            },
            {
                element: <ProjectLayout />,
                children: [
                    {
                        path: paths.project.datasets.index.pattern,
                        children: [
                            {
                                index: true,
                                element: <Datasets />,
                            },
                            {
                                path: paths.project.datasets.record.pattern,
                                element: <Record />,
                            },
                            {
                                path: paths.project.datasets.record_new.pattern,
                                element: <Record />,
                            },
                        ],
                    },
                    {
                        path: paths.project.models.index.pattern,
                        element: <Models />,
                    },
                    {
                        path: paths.project.robotConfiguration.index.pattern,
                        element: <RobotConfiguration />,
                    },
                    {
                        path: paths.project.cameras.index.pattern,
                        element: <CamerasLayout />,
                        children: [
                            {
                                index: true,
                                element: <div>Select a camera or create a new one</div>,
                            },
                            {
                                path: paths.project.cameras.new.pattern,
                                element: <div>New</div>,
                            },
                            {
                                path: paths.project.cameras.show.pattern,
                                element: <Camera />,
                            },
                            {
                                path: paths.project.cameras.overview.pattern,
                                element: <CameraOverview />,
                            },
                            {
                                path: paths.project.cameras.webcam.pattern,
                                element: <CameraWebcam />,
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
