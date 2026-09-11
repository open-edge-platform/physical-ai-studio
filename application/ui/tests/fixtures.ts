import { defineNetworkFixture, type NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';

import { handlers, http } from '../src/api/utils';
import { JobsDialogPage } from './pages/jobs-dialog-page';

interface Fixtures {
    network: NetworkFixture;
    jobsDialogPage: JobsDialogPage;
}

const test = testBase.extend<Fixtures>({
    network: [
        async ({ context }, use) => {
            const network = defineNetworkFixture({
                context,
                handlers: [...handlers],
            });

            await network.enable();
            await use(network);
            await network.disable();
        },
        { auto: true },
    ],
    jobsDialogPage: async ({ page }, use) => {
        await use(new JobsDialogPage(page));
    },
});

export { expect, http, test };
