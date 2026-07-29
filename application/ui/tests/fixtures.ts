import { defineNetworkFixture, type NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';

import { handlers, http } from '../src/api/utils';

interface Fixtures {
    network: NetworkFixture;
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
});

export { expect, http, test };
