// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpResponse } from 'msw';

import { getMockedTrainJob } from '../src/test-utils/mocks/mock-train-job';
import { expect, http, test } from './fixtures';

const PROJECT_A = 'project-1';
const PROJECT_B = 'project-2';

const jobA = getMockedTrainJob({
    id: 'job-a',
    project_id: PROJECT_A,
    status: 'running',
    payload: { model_name: 'project-a-model' } as never,
});
const jobB = getMockedTrainJob({
    id: 'job-b',
    project_id: PROJECT_B,
    status: 'completed',
    end_time: '2026-08-01T01:00:00Z',
    payload: { model_name: 'project-b-model' } as never,
});

test.describe('Jobs footer entry point', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/jobs', () => HttpResponse.json([jobA, jobB])),
            http.get('/api/logs/sources', () =>
                HttpResponse.json([{ id: 'application', name: 'Application', type: 'application', created_at: null }])
            ),
            http.get('/api/logs/{source_id}/stream', () => new HttpResponse('data: "DONE"\n\n' as never))
        );
    });

    test('scopes Jobs to the current project on a project subpage', async ({ jobsDialogPage }) => {
        await jobsDialogPage.gotoProjectModels(PROJECT_A);
        await jobsDialogPage.openJobs();

        await expect(jobsDialogPage.dialog.getByRole('heading', { name: 'Current project jobs' })).toBeVisible();
        await expect(jobsDialogPage.dialog.getByText('project-a-model')).toBeVisible();
        await expect(jobsDialogPage.dialog.getByText('project-b-model')).toBeHidden();
    });

    test("shows every project's jobs on the global Projects page", async ({ jobsDialogPage }) => {
        await jobsDialogPage.gotoGlobal();
        await jobsDialogPage.openJobs();

        await expect(jobsDialogPage.dialog.getByRole('heading', { name: 'All projects jobs' })).toBeVisible();
        await expect(jobsDialogPage.dialog.getByText('project-a-model')).toBeVisible();
        await expect(jobsDialogPage.dialog.getByText('project-b-model')).toBeVisible();
    });

    test(
        'opens on the "All jobs" tab, supports selecting another tab, and dismisses with the close (X) button, ' +
            'returning focus',
        async ({ jobsDialogPage }) => {
            await jobsDialogPage.gotoProjectModels(PROJECT_A);
            await jobsDialogPage.openJobs();

            await expect(jobsDialogPage.tab('All jobs')).toHaveAttribute('aria-selected', 'true');

            await jobsDialogPage.selectTab('Running jobs');
            await expect(jobsDialogPage.tab('Running jobs')).toHaveAttribute('aria-selected', 'true');

            await jobsDialogPage.closeJobs();

            await expect(jobsDialogPage.dialog).toBeHidden();
            await expect(jobsDialogPage.jobsButton).toBeFocused();
        }
    );

    test('opens fullscreen logs for a row and returns to Jobs with the selected tab preserved', async ({
        jobsDialogPage,
    }) => {
        await jobsDialogPage.gotoProjectModels(PROJECT_A);
        await jobsDialogPage.openJobs();

        await jobsDialogPage.selectTab('Running jobs');
        await expect(jobsDialogPage.tab('Running jobs')).toHaveAttribute('aria-selected', 'true');

        await jobsDialogPage.viewLogsFor('job-a');

        await expect(jobsDialogPage.logsHeading).toBeVisible();
        await expect(jobsDialogPage.dialog).toBeHidden();

        await jobsDialogPage.closeLogs();

        await expect(jobsDialogPage.dialog).toBeVisible();
        await expect(jobsDialogPage.tab('Running jobs')).toHaveAttribute('aria-selected', 'true');
    });
});
