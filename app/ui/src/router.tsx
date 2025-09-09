import { Suspense } from 'react';

import { Loading } from '@geti/ui';
import { redirect } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { ErrorMessage, ErrorPage } from './components/error-page/error-page';
import { Layout } from './layout';
import { Inference } from './routes/inference/inference';

const root = path('/');
const inference = root.path('/inference');
const robotConfiguration = root.path('/robot-configuration');
const datasets = root.path('/datasets');
const models = root.path('/models');

export const paths = {
    root,
    inference: {
        index: inference,
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
                index: true,
                loader: () => {
                    return redirect(paths.robotConfiguration.index({}));
                },
            },
            {
                path: paths.robotConfiguration.index.pattern,
                element: <ErrorMessage message={'Comming soon...'} />,
            },
            {
                path: paths.datasets.index.pattern,
                element: <ErrorMessage message={'Comming soon...'} />,
            },
            {
                path: paths.models.index.pattern,
                element: <ErrorMessage message={'Comming soon...'} />,
            },
            {
                path: '*',
                loader: () => {
                    return redirect(paths.robotConfiguration.index({}));
                },
            },
        ],
    },
]);
