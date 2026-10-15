import { describe, expect, it } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaVideoRecordingSettings } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { VideoRecordingSettingsForm } from './video-recording-settings-form';

const videoRecording: SchemaVideoRecordingSettings = { preset: null };

const renderForm = (settings: SchemaVideoRecordingSettings = videoRecording) => {
    server.use(
        http.get('/api/system/video-recording-presets', () =>
            HttpResponse.json({
                presets: [
                    {
                        id: 'intel-qsv-h264',
                        name: 'Intel GPU QSV',
                        description: 'Hardware H.264 encoding using Intel Quick Sync Video.',
                    },
                ],
            })
        )
    );
    return render(<VideoRecordingSettingsForm videoRecording={settings} />);
};

describe('VideoRecordingSettingsForm', () => {
    it('offers Automatic and the presets verified by the backend', async () => {
        renderForm();

        expect(await screen.findByRole('button', { name: /Video encoder/ })).toHaveTextContent('Automatic');
        await userEvent.setup().click(screen.getByRole('button', { name: /Video encoder/ }));
        expect(await screen.findByRole('option', { name: /Intel GPU QSV/ })).toBeInTheDocument();
    });

    it('saves an explicit backend preset', async () => {
        let requestBody: unknown;
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                requestBody = await request.json();
                return HttpResponse.json({
                    video_recording: { preset: 'intel-qsv-h264' },
                    trainer: {
                        request_timeout_s: 30,
                        download_read_timeout_s: 120,
                        stream_reconnect_max_s: 900,
                        stream_reconnect_backoff_max_s: 30,
                    },
                    huggingface: { hf_token: null },
                    ssh: {
                        connect_timeout_s: 10,
                        command_timeout_s: 15,
                        preflight_timeout_s: 30,
                        image_pull_timeout_s: 1800,
                        trainer_shm_size_gb: 32,
                    },
                    hotkeys: { bindings: {} },
                });
            })
        );
        const user = userEvent.setup();
        renderForm();

        await user.click(await screen.findByRole('button', { name: /Video encoder/ }));
        await user.click(await screen.findByRole('option', { name: /Intel GPU QSV/ }));
        await user.click(screen.getByRole('button', { name: 'Save' }));

        expect(requestBody).toEqual({ video_recording: { preset: 'intel-qsv-h264' } });
    });
});
