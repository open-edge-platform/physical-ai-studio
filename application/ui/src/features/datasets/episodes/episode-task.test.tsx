// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti-ui/ui';
import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { delay, HttpResponse } from 'msw';

import { $api } from '../../../api/client';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { getMockedEpisodeDetail } from '../../../test-utils/mocks/mock-episode-detail';
import { render } from '../../../test-utils/render';
import { EpisodeTask } from './episode-task';

const DATASET_ID = 'dataset-1';
const EPISODE_PATH = '/api/dataset/{dataset_id}/episodes/{episode_index}';

const episode = getMockedEpisodeDetail({ episode_index: 1, tasks: ['plcae the cube'] });
const episodeQueryKey = ['get', EPISODE_PATH, { params: { path: { dataset_id: DATASET_ID, episode_index: 1 } } }];
const episodesQueryKey = [
    'get',
    '/api/dataset/{dataset_id}/episodes',
    { params: { path: { dataset_id: DATASET_ID } } },
];

const renderEpisodeTask = (queryClient = createQueryClient()) => {
    const user = userEvent.setup();
    render(<EpisodeTask datasetId={DATASET_ID} episode={episode} />, { queryClient });
    return user;
};

const editTask = async (user: ReturnType<typeof userEvent.setup>, task: string) => {
    await user.click(await screen.findByRole('button', { name: 'Edit task' }));
    const field = screen.getByRole('textbox', { name: 'Episode task' });
    await user.clear(field);
    await user.type(field, task);
    return field;
};

describe('EpisodeTask', () => {
    it('shows the task with an edit button', async () => {
        renderEpisodeTask();

        expect(await screen.findByText('plcae the cube')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Edit task' })).toBeInTheDocument();
    });

    it('saves the trimmed task and refreshes the episode and the episode list', async () => {
        let receivedBody: unknown;
        let receivedPath: unknown;
        server.use(
            http.patch(EPISODE_PATH, async ({ request, params }) => {
                receivedBody = await request.json();
                receivedPath = params;
                return HttpResponse.json({ ...episode, tasks: ['place the cube'] });
            })
        );
        const queryClient = createQueryClient();
        queryClient.setQueryData(episodeQueryKey, episode);
        queryClient.setQueryData(episodesQueryKey, [episode]);
        const user = renderEpisodeTask(queryClient);

        await editTask(user, '  place the cube ');
        await user.click(screen.getByRole('button', { name: 'Save task' }));

        await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Episode task' })).not.toBeInTheDocument());
        expect(receivedBody).toEqual({ task: 'place the cube' });
        expect(receivedPath).toEqual({ dataset_id: DATASET_ID, episode_index: '1' });
        expect(queryClient.getQueryData(episodeQueryKey)).toEqual({ ...episode, tasks: ['place the cube'] });
        expect(queryClient.getQueryState(episodesQueryKey)?.isInvalidated).toBe(true);
    });

    it.each(['before', 'during'])('does not let a fetch started %s the save bring back the old task', async (when) => {
        let task = 'plcae the cube';
        let gets = 0;
        const queryClient = createQueryClient();
        const refetch = () => void queryClient.refetchQueries({ queryKey: episodeQueryKey });
        server.use(
            http.get(EPISODE_PATH, async () => {
                gets += 1;
                const served = task;
                if (gets === 2) {
                    await delay(300); // read before the save, answered after it
                }
                return HttpResponse.json({ ...episode, tasks: [served] });
            }),
            http.patch(EPISODE_PATH, async () => {
                if (when === 'during') {
                    refetch();
                    await delay(50);
                }
                task = 'place the cube';
                return HttpResponse.json({ ...episode, tasks: [task] });
            })
        );
        const Shown = () => {
            const { data } = $api.useQuery('get', EPISODE_PATH, {
                params: { path: { dataset_id: DATASET_ID, episode_index: 1 } },
            });
            return <span data-testid='shown'>{data?.tasks.join(', ')}</span>;
        };
        const user = userEvent.setup();
        render(
            <>
                <Shown />
                <EpisodeTask datasetId={DATASET_ID} episode={episode} />
            </>,
            { queryClient }
        );
        await waitFor(() => expect(screen.getByTestId('shown')).toHaveTextContent('plcae the cube'));

        await editTask(user, 'place the cube');
        if (when === 'before') {
            refetch();
        }
        await user.click(screen.getByRole('button', { name: 'Save task' }));

        await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Episode task' })).not.toBeInTheDocument());
        expect(screen.getByTestId('shown')).toHaveTextContent('place the cube');
        await delay(400);
        expect(screen.getByTestId('shown')).toHaveTextContent('place the cube');
    });

    it('shows the saved task when the refetch after saving fails', async () => {
        let saved = false;
        server.use(
            http.get(EPISODE_PATH, () =>
                saved ? new HttpResponse(null, { status: 500 }) : HttpResponse.json(episode)
            ),
            http.patch(EPISODE_PATH, () => {
                saved = true;
                return HttpResponse.json({ ...episode, tasks: ['place the cube'] });
            })
        );
        const Shown = () => {
            const { data } = $api.useQuery('get', EPISODE_PATH, {
                params: { path: { dataset_id: DATASET_ID, episode_index: 1 } },
            });
            return <span data-testid='shown'>{data?.tasks.join(', ')}</span>;
        };
        const user = userEvent.setup();
        render(
            <>
                <Shown />
                <EpisodeTask datasetId={DATASET_ID} episode={episode} />
            </>,
            { queryClient: createQueryClient() }
        );
        await waitFor(() => expect(screen.getByTestId('shown')).toHaveTextContent('plcae the cube'));

        await editTask(user, 'place the cube');
        await user.click(screen.getByRole('button', { name: 'Save task' }));

        await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Episode task' })).not.toBeInTheDocument());
        await waitFor(() => expect(screen.getByTestId('shown')).toHaveTextContent('place the cube'));
    });

    it('saves on Enter', async () => {
        let patched = false;
        server.use(
            http.patch(EPISODE_PATH, () => {
                patched = true;
                return HttpResponse.json({ ...episode, tasks: ['place the cube'] });
            })
        );
        const user = renderEpisodeTask();

        await editTask(user, 'place the cube{Enter}');

        await waitFor(() => expect(patched).toBe(true));
    });

    it('disables saving an empty task', async () => {
        const user = renderEpisodeTask();

        await editTask(user, '   ');

        expect(screen.getByRole('button', { name: 'Save task' })).toBeDisabled();
    });

    it('cancels without saving', async () => {
        let patched = false;
        server.use(
            http.patch(EPISODE_PATH, () => {
                patched = true;
                return HttpResponse.json(episode);
            })
        );
        const user = renderEpisodeTask();

        await editTask(user, 'place the cube');
        await user.click(screen.getByRole('button', { name: 'Cancel editing task' }));

        expect(screen.queryByRole('textbox', { name: 'Episode task' })).not.toBeInTheDocument();
        expect(screen.getByText('plcae the cube')).toBeInTheDocument();
        expect(patched).toBe(false);
    });

    it('cancels on Escape', async () => {
        const user = renderEpisodeTask();

        await editTask(user, 'place the cube{Escape}');

        expect(screen.queryByRole('textbox', { name: 'Episode task' })).not.toBeInTheDocument();
        expect(screen.getByText('plcae the cube')).toBeInTheDocument();
    });

    it('shows the error and keeps editing when saving fails', async () => {
        const notify = rs.spyOn(toast, 'negative').mockReturnValue(rs.fn());
        server.use(
            // Studio's own error body, which the generated spec does not describe.
            http.patch(EPISODE_PATH, ({ response }) =>
                response.untyped(
                    HttpResponse.json(
                        {
                            error_code: 'Dataset_in_use',
                            message: "Dataset 'Cubes' is loaded in a runtime session. Unload it there, then try again.",
                            http_status: 409,
                        },
                        { status: 409 }
                    )
                )
            )
        );
        const user = renderEpisodeTask();

        await editTask(user, 'place the cube');
        await user.click(screen.getByRole('button', { name: 'Save task' }));

        await waitFor(() =>
            expect(notify).toHaveBeenCalledWith(
                "Dataset 'Cubes' is loaded in a runtime session. Unload it there, then try again."
            )
        );
        expect(screen.getByRole('textbox', { name: 'Episode task' })).toHaveValue('place the cube');
        notify.mockRestore();
    });
});
