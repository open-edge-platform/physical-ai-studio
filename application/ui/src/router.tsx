import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { Outlet, redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ErrorPage } from './components/error-page/error-page';
import { Camera, CameraOverview } from './routes/cameras/camera';
import { Edit as CameraEdit } from './routes/cameras/edit';
import { Layout as CamerasLayout } from './routes/cameras/layout';
import { New as CamerasNew } from './routes/cameras/new';
import { CameraWebcam } from './routes/cameras/webcam';
import { Index as Datasets } from './routes/datasets/index';
import { Index as Models } from './routes/models/index';
import { Index as Inference } from './routes/models/inference/index';
import { OpenApi } from './routes/openapi';
import { Index as Projects } from './routes/projects/index';
import { ProjectLayout } from './routes/projects/project.layout';
import { Calibration } from './routes/robots/calibration';
import { Controller } from './routes/robots/controller';
import { Edit as RobotEdit } from './routes/robots/edit';
import { Layout as RobotsLayout } from './routes/robots/layout';
import { New as RobotsNew } from './routes/robots/new';
import { Robot } from './routes/robots/robot';
import { SetupMotors } from './routes/robots/setup-motors';
import { TabNavigation as RobotsTabNavigation } from './routes/robots/tab-navigation';

const root = path('/');
const projects = root.path('/projects');
const project = root.path('/projects/:project_id');
const robots = project.path('robots');
const robot = robots.path(':robot_id');
const datasets = project.path('/datasets');
const models = project.path('/models');
const cameras = project.path('cameras');
const environments = project.path('environments');

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
        },
        robots: {
            index: robots,
            new: robots.path('new'),
            edit: robot.path('edit'),
            show: robot,
            controller: robot.path('/controller'),
            calibration: robot.path('/calibration'),
            setupMotors: robot.path('/setup-motors'),
        },
        cameras: {
            index: cameras,
            webcam: cameras.path('/webcam'),
            overview: cameras.path('/overview'),
            new: cameras.path('/new'),
            edit: cameras.path(':camera_id/edit'),
            show: cameras.path(':camera_id'),
        },
        environments: {
            index: environments,
        },
        models: {
            index: models,
            inference: models.path('/:model_id/inference'),
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
                ],
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
                        element: <Datasets />,
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
                            <>
                                <RobotsTabNavigation />
                                <Outlet />
                            </>
                        ),
                        children: [
                            // Robots
                            {
                                path: paths.project.robots.new.pattern,
                                element: <RobotsNew />,
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
                                        element: <div>Illustration to persuade user to select robot</div>,
                                    },
                                    {
                                        path: paths.project.robots.show.pattern,
                                        element: <Robot />,
                                        children: [
                                            {
                                                index: true,
                                                loader: ({ params }) => {
                                                    return redirect(
                                                        paths.project.robots.controller({
                                                            project_id: params.project_id ?? '',
                                                            robot_id: params.robot_id ?? '',
                                                        })
                                                    );
                                                },
                                            },
                                            {
                                                path: paths.project.robots.controller.pattern,
                                                element: <Controller />,
                                            },
                                            {
                                                path: paths.project.robots.calibration.pattern,
                                                element: <Calibration />,
                                            },
                                            {
                                                path: paths.project.robots.setupMotors.pattern,
                                                element: <SetupMotors />,
                                            },
                                        ],
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
                                        element: <div>Select a camera or create a new one</div>,
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
