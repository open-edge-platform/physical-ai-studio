import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { SshSettingsForm } from './ssh-settings-form';

const baseSsh = {
    connect_timeout_s: 10,
    command_timeout_s: 15,
    preflight_timeout_s: 30,
    image_pull_timeout_s: 1800,
    readiness_timeout_s: 120,
    gpu_wait_giveup_s: 1800,
    min_free_disk_bytes: 53687091200,
};

describe('SshSettingsForm', () => {
    it('renders existing values in the timeout fields', () => {
        render(<SshSettingsForm ssh={{ ...baseSsh, connect_timeout_s: 42 }} />);

        expect(screen.getByRole('textbox', { name: /connect timeout/i })).toHaveValue('42');
    });

    it('saves a changed timeout', async () => {
        let patchedBody: unknown;
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                patchedBody = await request.json();
                return HttpResponse.json({
                    trainer: {
                        request_timeout_s: 30,
                        download_read_timeout_s: 120,
                        stream_reconnect_max_s: 900,
                        stream_reconnect_backoff_max_s: 30,
                    },
                    huggingface: { hf_token: null },
                    ssh: { ...baseSsh, connect_timeout_s: 42 },
                });
            })
        );

        const user = userEvent.setup();
        render(<SshSettingsForm ssh={baseSsh} />);

        const connectTimeoutField = screen.getByRole('textbox', { name: /connect timeout/i });
        await user.clear(connectTimeoutField);
        await user.type(connectTimeoutField, '42');
        await user.click(screen.getByRole('button', { name: /save/i }));

        expect(await screen.findByText('Saved')).toBeInTheDocument();
        expect(patchedBody).toMatchObject({ ssh: { connect_timeout_s: 42 } });
    });
});
