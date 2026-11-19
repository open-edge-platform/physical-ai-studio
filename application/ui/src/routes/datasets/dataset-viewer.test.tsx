import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { getMockedDataset } from '../../test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../../test-utils/mocks/mock-environment';
import { getMockedEpisode } from '../../test-utils/mocks/mock-episode';
import { getMockedEpisodeDetail } from '../../test-utils/mocks/mock-episode-detail';
import { getMockedRecordingEnvironment } from '../../test-utils/mocks/mock-recording-environment';
import { render } from '../../test-utils/render';
import { DatasetProvider } from './dataset-provider';
import { DatasetViewer } from './dataset-viewer';

const renderViewer = () =>
    render(
        <DatasetProvider dataset_id='dataset-1'>
            <DatasetViewer />
        </DatasetProvider>,
        {
            route: '/projects/project-1/datasets/dataset-1/episodes/0',
            path: '/projects/:project_id/datasets/:dataset_id/episodes/:episode_index',
        }
    );

describe('DatasetViewer recording environments', () => {
    beforeEach(() => {
        rs.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
        rs.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
        server.use(
            http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset())),
            http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([getMockedEpisode()])),
            http.get('/api/dataset/{dataset_id}/episodes/{episode_index}', () =>
                HttpResponse.json(getMockedEpisodeDetail())
            ),
            http.get('/api/dataset/{dataset_id}/recording-environments', () => HttpResponse.json([]))
        );
    });

    afterEach(() => {
        rs.restoreAllMocks();
    });

    it('keeps videos, playback controls and the timeline available without any environment', async () => {
        const user = userEvent.setup();
        const { container } = renderViewer();

        expect(await screen.findByRole('button', { name: 'Play' })).toBeInTheDocument();
        await waitFor(() => expect(container.querySelectorAll('video')).toHaveLength(2));
        await user.click(screen.getByRole('button', { name: 'Timeline' }));
        expect(screen.getByRole('button', { name: 'Timeline' })).toHaveAttribute('aria-expanded', 'true');
        await user.click(screen.getByRole('button', { name: 'Play' }));
        expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();
    });

    it('loads the associated environment rather than a project-list default', async () => {
        const environmentRequest = rs.fn();
        server.use(
            http.get('/api/dataset/{dataset_id}/recording-environments', () =>
                HttpResponse.json([getMockedRecordingEnvironment({ environment_id: 'environment-2' })])
            ),
            http.get('/api/projects/{project_id}/environments/{environment_id}', ({ params }) => {
                environmentRequest(params.environment_id);
                return HttpResponse.json(getMockedEnvironment({ id: 'environment-2', robots: [] }));
            })
        );
        renderViewer();

        await waitFor(() => expect(environmentRequest).toHaveBeenCalledWith('environment-2'));
        expect(await screen.findByRole('button', { name: 'Play' })).toBeInTheDocument();
    });
});
