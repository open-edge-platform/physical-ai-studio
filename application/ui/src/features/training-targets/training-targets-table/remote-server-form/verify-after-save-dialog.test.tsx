import { ToastQueue } from '@geti-ui/ui';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaRemoteServer } from '../../../../api/openapi-spec';
import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { render } from '../../../../test-utils/render';
import { VerifyAfterSaveDialog } from './verify-after-save-dialog';

const savedServer = {
    id: 'server-1',
    name: 'lab-gpu-box',
    ssh_host_alias: 'gpu-box',
    device_type: 'cuda',
    last_check_status: 'unknown',
} as SchemaRemoteServer;

describe('VerifyAfterSaveDialog', () => {
    it('closes immediately and still shows a toast when the background verification fails', async () => {
        // Regression test: `close()` unmounts this dialog synchronously, which
        // tears down its mutation observer. Callbacks passed to `mutate(...)`
        // are delivered through that observer and are silently dropped once it
        // has no listeners, so the failure toast must come from `mutateAsync`'s
        // own promise instead, which keeps settling after unmount.
        const user = userEvent.setup();
        const close = vi.fn();
        const toast = vi.spyOn(ToastQueue, 'negative').mockImplementation(() => () => {});
        server.use(
            http.post('/api/remote-servers/{remote_server_id}/check', () =>
                HttpResponse.json({ detail: [] } as never, { status: 500 })
            )
        );

        render(<VerifyAfterSaveDialog savedServer={savedServer} close={close} />);

        await user.click(screen.getByRole('button', { name: 'Pull & verify image' }));

        expect(close).toHaveBeenCalled();
        await vi.waitFor(() => expect(toast).toHaveBeenCalledWith(expect.stringMatching(/verification failed/i)));
    });

    it('closes without starting verification when skipped', async () => {
        const user = userEvent.setup();
        const close = vi.fn();

        render(<VerifyAfterSaveDialog savedServer={savedServer} close={close} />);
        await user.click(screen.getByRole('button', { name: 'Skip for now' }));

        expect(close).toHaveBeenCalled();
    });
});
