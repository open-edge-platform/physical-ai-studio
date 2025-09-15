import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { Outlet, redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ErrorPage } from './components/error-page/error-page';
import { Layout } from './layout';
import { Index as Datasets } from './routes/datasets/index';
import { Record } from './routes/datasets/record/record';
import { Index as Models } from './routes/models/index';
import { Index as Projects } from './routes/projects/index';
import { NewProject } from './routes/projects/new/new';
import { ProjectLayout } from './routes/projects/project.layout';
import { Index as RobotConfiguration } from './routes/robot-configuration/index';

const root = path('/');
const projects = root.path('/projects');
const project = root.path('/project/:project_id');
const inference = root.path('/inference');
const robotConfiguration = project.path('/robot-configuration');
const datasets = project.path('/datasets');
const models = project.path('/models');

export const paths = {
    root,
    inference: {
        index: inference,
    },
    projects: {
        index: projects,
        new: projects.path('/new'),
    },
    project: {
        datasets: {
            index: datasets,
            record: datasets.path('/record'),
        },
        robotConfiguration,
        models,
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
                path: paths.projects.index.pattern,
                element: <Layout />,
                children: [
                    {
                        index: true,
                        element: <Projects />,
                    },
                    {
                        path: paths.projects.new.pattern,
                        element: <NewProject />,
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
                        ],
                    },
                    {
                        path: paths.project.models.pattern,
                        element: <Models />,
                    },
                    {
                        path: paths.project.robotConfiguration.pattern,
                        element: <RobotConfiguration />,
                    },
                ],
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
