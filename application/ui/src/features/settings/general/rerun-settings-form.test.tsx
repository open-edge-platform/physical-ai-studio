import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedRerunSettings, getMockedUserSettings } from '../../../test-utils/mocks/mock-settings';
import { render } from '../../../test-utils/render';
import { RerunSettingsForm } from './rerun-settings-form';

describe('RerunSettingsForm', () => {
    it('keeps edits made during a save dirty until they are saved separately', async () => {
        let finishSave: () => void = () => {};
        const pending = new Promise<void>((resolve) => {
            finishSave = resolve;
        });
        const bodies: unknown[] = [];
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                bodies.push(await request.json());
                if (bodies.length === 1) await pending;
                return HttpResponse.json(getMockedUserSettings());
            })
        );
        const user = userEvent.setup();
        render(<RerunSettingsForm rerun={getMockedRerunSettings()} />);
        await user.click(screen.getByRole('checkbox', { name: 'Open the native viewer when a model run starts' }));
        await user.click(screen.getByRole('button', { name: 'Save' }));
        await waitFor(() => expect(bodies).toHaveLength(1));
        await user.click(screen.getByRole('checkbox', { name: 'Enable Rerun logging' }));
        finishSave();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled());
        expect(screen.queryByText('Saved')).not.toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Save' }));
        expect(await screen.findByText('Saved')).toBeInTheDocument();
        expect(bodies).toEqual([
            { rerun: getMockedRerunSettings({ open_viewer: true }) },
            { rerun: getMockedRerunSettings({ open_viewer: true, enabled: false }) },
        ]);
    });

    it('defaults to background logging without opening a native window', () => {
        render(<RerunSettingsForm rerun={getMockedRerunSettings()} />);

        expect(screen.getByRole('checkbox', { name: 'Enable Rerun logging' })).toBeChecked();
        expect(
            screen.getByRole('checkbox', { name: 'Open the native viewer when a model run starts' })
        ).not.toBeChecked();
        expect(screen.getByRole('textbox', { name: /Local Rerun port/ })).toHaveValue('9,876');
        expect(screen.getByText(/Changes apply to new runs/)).toBeInTheDocument();
    });

    it('saves opting into the native viewer', async () => {
        let body: unknown;
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                body = await request.json();
                return HttpResponse.json(getMockedUserSettings());
            })
        );
        const user = userEvent.setup();
        render(<RerunSettingsForm rerun={getMockedRerunSettings()} />);
        await user.click(screen.getByRole('checkbox', { name: 'Open the native viewer when a model run starts' }));
        await user.click(screen.getByRole('button', { name: 'Save' }));
        expect(await screen.findByText('Saved')).toBeInTheDocument();
        expect(body).toEqual({ rerun: getMockedRerunSettings({ open_viewer: true }) });
    });

    it('switches to connect mode and saves the address', async () => {
        let body: unknown;
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                body = await request.json();
                return HttpResponse.json(getMockedUserSettings());
            })
        );
        const user = userEvent.setup();
        render(<RerunSettingsForm rerun={getMockedRerunSettings()} />);
        await user.click(screen.getByRole('button', { name: /Rerun mode/ }));
        await user.click(await screen.findByRole('option', { name: 'Connect to an existing server' }));
        expect(screen.queryByRole('textbox', { name: /Local Rerun port/ })).not.toBeInTheDocument();
        const address = screen.getByRole('textbox', { name: 'Rerun connection address' });
        await user.clear(address);
        await user.type(address, 'remote:1234');
        await user.click(screen.getByRole('button', { name: 'Save' }));
        expect(await screen.findByText('Saved')).toBeInTheDocument();
        expect(body).toEqual({ rerun: getMockedRerunSettings({ mode: 'connect', connect_addr: 'remote:1234' }) });
    });

    it('can disable Rerun logging', async () => {
        let body: unknown;
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                body = await request.json();
                return HttpResponse.json(getMockedUserSettings());
            })
        );
        const user = userEvent.setup();
        render(<RerunSettingsForm rerun={getMockedRerunSettings()} />);
        await user.click(screen.getByRole('checkbox', { name: 'Enable Rerun logging' }));
        expect(screen.getByRole('button', { name: /Rerun mode/ })).toBeDisabled();
        await user.click(screen.getByRole('button', { name: 'Save' }));
        expect(await screen.findByText('Saved')).toBeInTheDocument();
        expect(body).toEqual({ rerun: getMockedRerunSettings({ enabled: false }) });
    });
});
