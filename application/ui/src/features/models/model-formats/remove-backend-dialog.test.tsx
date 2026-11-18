import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { render } from '../../../test-utils/render';
import { RemoveBackendDialog } from './remove-backend-dialog';

describe('RemoveBackendDialog', () => {
    it('confirms removal of only the selected backend', async () => {
        const removed = rs.fn();
        const close = rs.fn();
        const model = getMockedModel();
        server.use(
            http.delete('/api/models/{model_id}/exports/{backend}', ({ params }) => {
                removed(params);
                return new HttpResponse(null, { status: 204 });
            })
        );
        const user = userEvent.setup();
        render(<RemoveBackendDialog model={model} backend='torch' close={close} />);
        expect(screen.getByText(/checkpoint, training history, and other backends are kept/i)).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Remove model' }));
        await waitFor(() => expect(removed).toHaveBeenCalledWith({ model_id: model.id, backend: 'torch' }));
        await waitFor(() => expect(close).toHaveBeenCalled());
    });

    it('keeps the dialog open when the export is in use', async () => {
        const close = rs.fn();
        server.use(
            http.delete('/api/models/{model_id}/exports/{backend}', () =>
                HttpResponse.json(
                    { detail: [{ loc: [], msg: 'Unload the model first', type: 'conflict' }] },
                    { status: 409 }
                )
            )
        );
        const user = userEvent.setup();
        render(<RemoveBackendDialog model={getMockedModel()} backend='torch' close={close} />);
        await user.click(screen.getByRole('button', { name: 'Remove model' }));
        expect(await screen.findByRole('alert')).toHaveTextContent('Unload the model first');
        expect(close).not.toHaveBeenCalled();
    });
});
