import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it, vi } from 'vitest';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { RobotFormProvider } from './provider';
import { RobotCatalogDialog } from './robot-catalog-dialog';

const catalogEntries = [
    {
        type: 'SO101_Follower',
        display_name: 'SO101 Follower',
        category: 'SO101',
        source: 'internal' as const,
        role: 'follower' as const,
        preview_thumbnail: null,
        urdf_path: '/api/robots/catalog/SO101_Follower/urdf',
        package_map: { SO101: '/api/robots/catalog/SO101_Follower' },
        joint_map: {},
    },
];

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

describe('RobotCatalogDialog', () => {
    it('shows robots from plugins that are not installed yet', async () => {
        server.use(
            http.get('/api/robots/catalog', () => HttpResponse.json(catalogEntries)),
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin]))
        );

        render(
            <RobotFormProvider>
                <RobotCatalogDialog close={vi.fn()} />
            </RobotFormProvider>,
            { route: '/plugins', path: '/plugins' }
        );

        expect(await screen.findByRole('heading', { name: 'SO101' })).toBeVisible();
        expect(await screen.findByRole('heading', { name: 'Available plugins' })).toBeVisible();
        expect(screen.getByText('MuJoCo SO101 Follower')).toBeVisible();
        expect(screen.getByText('Not installed')).toBeVisible();
    });

    it('opens a plugin-specific install modal when installing from the picker', async () => {
        server.use(
            http.get('/api/robots/catalog', () => HttpResponse.json(catalogEntries)),
            http.get('/api/plugins', () => HttpResponse.json([availablePlugin]))
        );
        const close = vi.fn();
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <RobotCatalogDialog close={close} />
            </RobotFormProvider>,
            { route: '/plugins', path: '/plugins' }
        );

        await user.click(await screen.findByRole('button', { name: 'Install plugin' }));
        expect(close).not.toHaveBeenCalled();
        expect(await screen.findByRole('heading', { name: 'MuJoCo Plugin' })).toBeVisible();
        expect(screen.getAllByText('MuJoCo-backed SO-101 simulation integration.')).toHaveLength(2);
        expect(screen.getAllByText('MuJoCo SO101 Follower')).toHaveLength(2);
        expect(screen.getByRole('button', { name: 'Install' })).toBeVisible();
    });
});
