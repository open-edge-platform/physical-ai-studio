import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it, vi } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { PluginsView } from './plugins';

const installedPlugin = {
    id: 'physicalai-rebot-b601-plugin',
    name: 'ReBot Plugin',
    description: 'ReBot B601 and Arm102 robot integrations.',
    repo_url: 'https://github.com/example/rebot',
    installed: true,
    installed_version: '0.1.0',
    robot_count: 1,
};

const availablePlugin = {
    id: 'physicalai-mujoco-so101-plugin',
    name: 'MuJoCo Plugin',
    description: 'MuJoCo-backed SO-101 simulation integration.',
    repo_url: 'https://github.com/example/mujoco',
    installed: false,
    installed_version: null,
    robot_count: 1,
};

const lerobotPlugin = {
    id: 'physicalai-lerobot-plugin',
    name: 'LeRobot Plugin',
    description: 'Robot and teleoperator configurations discovered from LeRobot.',
    repo_url: 'https://github.com/example/lerobot',
    installed: true,
    installed_version: '0.1.0',
    robot_count: 0,
};

describe('PluginsView', () => {
    it('renders installed and available plugins in a single table', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([installedPlugin, availablePlugin, lerobotPlugin]))
        );

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await screen.findByTestId('plugin-row-physicalai-rebot-b601-plugin');

        expect(await screen.findByRole('heading', { name: 'Plugins' })).toBeVisible();
        expect(screen.getByText('Plugin')).toBeVisible();
        expect(screen.getByText('ReBot Plugin')).toBeVisible();
        expect(screen.getByText('MuJoCo Plugin')).toBeVisible();
    });

    it('opens and dismisses a restart prompt after installing a plugin', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin])),
            http.post('/api/plugins', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Install' }));

        expect(await screen.findByText('Plugin changes require a server restart to become active.')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Restart now' })).toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Later' }));
        expect(screen.queryByText('Plugin changes require a server restart to become active.')).not.toBeInTheDocument();
    });

    it('restarts after confirming the restart prompt', async () => {
        let restartCalls = 0;
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin])),
            http.post('/api/plugins', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([])),
            http.post('/api/system/restart', () => {
                restartCalls += 1;
                return HttpResponse.json({ status: 'restarting' });
            }),
            http.get('/api/health', () => {
                return HttpResponse.json({
                    status: 'healthy',
                    instance_id: restartCalls === 0 ? 'before-restart' : 'after-restart',
                    restart_required: restartCalls === 0,
                });
            })
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Install' }));
        await user.click(await screen.findByRole('button', { name: 'Restart now' }));

        await vi.waitFor(() => {
            expect(restartCalls).toBe(1);
        });

        expect(await screen.findByText('Waiting for server restart…')).toBeVisible();
    });

    it('allows uninstall for plugins with robots in use', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([{ ...installedPlugin, robot_count: 1 }])),
            http.delete('/api/plugins/{plugin_id}', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Uninstall' }));
        expect(await screen.findByText('Plugin changes require a server restart to become active.')).toBeVisible();
    });

    it('opens the restart prompt after uninstalling a plugin', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([installedPlugin])),
            http.delete('/api/plugins/{plugin_id}', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Uninstall' }));

        expect(await screen.findByText('Plugin changes require a server restart to become active.')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Restart now' })).toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Later' }));
    });
});
