import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ErrorPage } from './components/error-page/error-page';
import { Layout } from './layout';
import { Index as Datasets } from './routes/datasets/index';
import { Index as Models } from './routes/models/index';
import { Index as Projects } from './routes/projects/index';
import { NewProject } from './routes/projects/new/new';
import { Index as RobotConfiguration } from './routes/robot-configuration/index';

const root = path('/');
const projects = root.path('/projects');
const inference = root.path('/inference');
const robotConfiguration = root.path('/robot-configuration');
const datasets = root.path('/datasets');
const models = root.path('/models');

export const paths = {
    root,
    inference: {
        index: inference,
    },
    projects: {
        index: projects,
        new: projects.path('/new'),
        edit: projects.path('/edit/:projectId'),
    },
    robotConfiguration: {
        index: robotConfiguration,
        controller: robotConfiguration.path('/controller'),
        calibration: robotConfiguration.path('/calibration'),
        setupMotors: robotConfiguration.path('/setup-motors'),
    },
    datasets: {
        index: datasets,
        show: datasets.path('/:datasetId'),
    },
    models: {
        index: models,
    },
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: (
            <Suspense fallback={<Loading mode='fullscreen' />}>
                <Layout />
            </Suspense>
        ),
        errorElement: <ErrorPage />,
        children: [
            {
                path: paths.robotConfiguration.index.pattern,
                element: <RobotConfiguration />,
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
                        element: <NewProject />,
                    },
                    {
                        path: paths.projects.edit.pattern,
                        element: <>Edit</>,
                    },
                ],
            },
            {
                path: paths.datasets.index.pattern,
                element: <Datasets />,
            },
            {
                path: paths.models.index.pattern,
                element: <Models />,
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
