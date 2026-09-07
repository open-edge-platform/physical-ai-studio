import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { Plugins } from './index';

const PLUGINS_PATH = '/api/plugins';

describe('Plugins route', () => {
    beforeEach(() => {
        server.use(http.get(PLUGINS_PATH, () => HttpResponse.json([])));
    });

    afterEach(() => {
        delete process.env.PUBLIC_ENABLE_PLUGINS;
    });

    it('renders the plugin management page when the feature is enabled', async () => {
        process.env.PUBLIC_ENABLE_PLUGINS = 'true';

        render(<Plugins />, { route: '/plugins', path: '/plugins' });

        expect(await screen.findByText('Install and manage plugins for the server.')).toBeVisible();
    });

    it('renders a disabled page when the feature is disabled', () => {
        process.env.PUBLIC_ENABLE_PLUGINS = 'false';

        render(<Plugins />, { route: '/plugins', path: '/plugins' });

        expect(screen.getByText('Plugin management is disabled for this instance.')).toBeVisible();
        expect(screen.queryByText('Install and manage plugins for the server.')).not.toBeInTheDocument();
    });
});
