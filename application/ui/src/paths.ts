import { path } from 'static-path';

const root = path('/');
const settings = root.path('/settings');
const plugins = root.path('/plugins');
const projects = root.path('/projects');
const project = root.path('/projects/:project_id');
const robots = project.path('robots');
const robot = robots.path(':robot_id');
const datasets = project.path('/datasets');
const dataset = datasets.path(':dataset_id');
const datasetEpisode = dataset.path('episodes').path(':episode_index');
const models = project.path('/models');
const cameras = project.path('cameras');
const environments = project.path('environments');
const environment = environments.path(':environment_id');

export const paths = {
    root,
    settings: {
        index: settings,
        trainingTargets: settings.path('/training-targets'),
        hotkeys: settings.path('/hotkeys'),
        storage: settings.path('/storage'),
    },
    openapi: root.path('/openapi'),
    plugins: {
        index: plugins,
    },
    projects: {
        index: projects,
    },
    project: {
        index: project,
        datasets: {
            index: datasets,
            show: dataset,
            showEpisode: datasetEpisode,
            record: dataset.path('record'),
        },
        robots: {
            index: robots,
            new: robots.path('new'),
            so101Setup: robots.path('new/so101-setup'),
            rebotRsSetup: robots.path('new/rebot-rs-setup'),
            stararmSetup: robots.path('new/stararm-setup'),
            zeroCalibration: robots.path('new/zero-calibration'),
            edit: robot.path('edit'),
            show: robot,
        },
        cameras: {
            index: cameras,
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
            inference: models.path('/:model_id/inference/:backend'),
        },
    },
};
