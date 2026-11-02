// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

export class JobsDialogPage {
    constructor(private page: Page) {}

    async gotoProjectModels(projectId: string) {
        await this.page.goto(`/projects/${projectId}/models`);
    }

    async gotoGlobal() {
        await this.page.goto('/projects');
    }

    get jobsButton() {
        return this.page.getByRole('button', { name: 'Jobs' });
    }

    get dialog() {
        return this.page.getByRole('dialog', { name: /jobs$/i });
    }

    get closeButton() {
        return this.dialog.getByRole('button', { name: 'Dismiss' });
    }

    async openJobs() {
        await this.jobsButton.click();
        await this.dialog.waitFor();
    }

    async closeJobs() {
        await this.closeButton.click();
    }

    tab(label: string) {
        return this.dialog.getByRole('tab', { name: new RegExp(label, 'i') });
    }

    async selectTab(label: string) {
        await this.tab(label).click();
    }

    row(jobId: string) {
        return this.dialog.getByTestId(jobId);
    }

    async openRowMenu(jobId: string) {
        await this.row(jobId).getByRole('button', { name: 'Job options' }).click();
    }

    async viewLogsFor(jobId: string) {
        await this.openRowMenu(jobId);
        await this.page.getByRole('menuitem', { name: 'Logs' }).click();
    }

    get logsHeading() {
        return this.page.getByRole('heading', { name: 'Logs' });
    }

    async closeLogs() {
        await this.page.getByRole('button', { name: 'Close' }).click();
    }
}
