import { screen, within } from '@testing-library/react';
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
    category: 'ReBot',
    source: 'first_party' as const,
    repo_url: 'https://github.com/example/rebot',
    installed: true,
    installed_version: '0.1.0',
    in_use_robot_count: 0,
    robots: [
        {
            type: 'ReBot_B601_DM_Follower',
            display_name: 'ReBot B601 DM Follower',
            role: 'follower' as const,
            installed: true,
        },
    ],
};

const availablePlugin = {
    id: 'physicalai-mujoco-so101-plugin',
    name: 'MuJoCo Plugin',
    description: 'MuJoCo-backed SO-101 simulation integration.',
    category: 'MuJoCo',
    source: 'first_party' as const,
    repo_url: 'https://github.com/example/mujoco',
    installed: false,
    installed_version: null,
    in_use_robot_count: 0,
    robots: [
        {
            type: 'MuJoCo_SO101_Follower',
            display_name: 'MuJoCo SO101 Follower',
            role: 'follower' as const,
            installed: false,
        },
    ],
};

const lerobotPlugin = {
    id: 'physicalai-lerobot-plugin',
    name: 'LeRobot Plugin',
    description: 'Robot and teleoperator configurations discovered from LeRobot.',
    category: 'LeRobot',
    source: 'first_party' as const,
    repo_url: 'https://github.com/example/lerobot',
    installed: true,
    installed_version: '0.1.0',
    in_use_robot_count: 0,
    robots: [],
    extensions: [
        {
            id: 'lerobot-teleoperator-spacemouse',
            name: 'SpaceMouse Teleoperator',
            description: 'Adds the LeRobot SpaceMouse leader teleoperator.',
            repo_url: null,
            installed: false,
            installed_version: null,
        },
    ],
};

describe('PluginsView', () => {
    it('renders installed and available plugins in a single table', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([installedPlugin, availablePlugin, lerobotPlugin]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        const rebotRow = await screen.findByTestId('plugin-row-physicalai-rebot-b601-plugin');
        await user.click(within(rebotRow).getByText('ReBot Plugin'));

        expect(await screen.findByRole('heading', { name: 'Plugins' })).toBeVisible();
        expect(screen.getByText('Plugin')).toBeVisible();
        expect(screen.getAllByText('Robots')).toHaveLength(2);
        expect(screen.getByText('ReBot Plugin')).toBeVisible();
        expect(screen.getByText('MuJoCo Plugin')).toBeVisible();
        expect(screen.getByRole('heading', { name: 'Robots' })).toBeVisible();
        expect(screen.getByText('ReBot B601 DM Follower')).toBeVisible();
        expect(screen.getAllByText('1 robot')).toHaveLength(2);
    });

    it('opens a restart prompt after installing a plugin', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin])),
            http.post('/api/plugins/{plugin_id}/install', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        await user.click(await screen.findByRole('button', { name: 'Install' }));

        expect(await screen.findByText('Plugin changes require a server restart to become active.')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Restart now' })).toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Later' }));
    });

    it('restarts after confirming the restart prompt', async () => {
        let healthCalls = 0;
        let restartCalls = 0;
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin])),
            http.post('/api/plugins/{plugin_id}/install', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([])),
            http.post('/api/system/restart', () => {
                restartCalls += 1;
                return HttpResponse.json({ status: 'restarting' });
            }),
            http.get('/api/health', () => {
                healthCalls += 1;
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

    it('shows extensions for an installed plugin and lets the user install them', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([lerobotPlugin])),
            http.post('/api/plugins/{plugin_id}/install', () => HttpResponse.json({ restart_required: true })),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        const lerobotRow = await screen.findByTestId('plugin-row-physicalai-lerobot-plugin');
        await user.click(within(lerobotRow).getByText('LeRobot Plugin'));

        expect(await screen.findByRole('heading', { name: 'Extensions' })).toBeVisible();
        expect(screen.getByText('SpaceMouse Teleoperator')).toBeVisible();

        await user.click(screen.getByRole('button', { name: 'Install' }));
        expect(await screen.findByText('Plugin changes require a server restart to become active.')).toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Later' }));
    });

    it('disables uninstall for plugins with robots in use', async () => {
        server.use(http.get('/api/plugins', () => HttpResponse.json([{ ...installedPlugin, in_use_robot_count: 2 }])));
        const user = userEvent.setup();

        render(<PluginsView />, { route: '/plugins', path: '/plugins' });

        const rebotRow = await screen.findByTestId('plugin-row-physicalai-rebot-b601-plugin');
        await user.click(within(rebotRow).getByText('ReBot Plugin'));

        const uninstallButton = await screen.findByRole('button', { name: 'Uninstall' });
        expect(uninstallButton).toBeDisabled();
        expect(screen.getByText(/In use by 2 robots/)).toBeVisible();
    });

    it('opens the restart prompt after uninstalling a plugin', async () => {
        server.use(
            http.get('/api/plugins', () => HttpResponse.json([installedPlugin])),
            http.post('/api/plugins/{plugin_id}/uninstall', () => HttpResponse.json({ restart_required: true })),
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
