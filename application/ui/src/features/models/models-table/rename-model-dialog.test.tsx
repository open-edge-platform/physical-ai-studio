import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { render } from '../../../test-utils/render';
import { RenameModelDialog } from './rename-model-dialog';

const model = getMockedModel();

describe('RenameModelDialog', () => {
    it('saves the trimmed name, invalidates the model queries, and closes', async () => {
        const user = userEvent.setup();
        const close = rs.fn();
        let receivedBody: unknown;
        let receivedId: string | undefined;
        server.use(
            http.patch('/api/models/{model_id}', async ({ request, params }) => {
                receivedBody = await request.json();
                receivedId = params.model_id;
                return HttpResponse.json(getMockedModel({ name: 'Renamed' }));
            })
        );
        const queryClient = createQueryClient();
        const modelKey = ['get', '/api/models/{model_id}', { params: { path: { model_id: model.id } } }] as const;
        const listKey = [
            'get',
            '/api/projects/{project_id}/models',
            { params: { path: { project_id: model.project_id } } },
        ] as const;
        queryClient.setQueryData(modelKey, { model });
        queryClient.setQueryData(listKey, [model]);
        render(<RenameModelDialog model={model} close={close} />, { queryClient });

        const name = screen.getByRole('textbox', { name: /Model name/ });
        expect(name).toHaveValue(model.name);
        expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
        await user.clear(name);
        await user.type(name, '  Renamed  ');
        await user.click(screen.getByRole('button', { name: 'Save' }));

        await waitFor(() => expect(close).toHaveBeenCalledOnce());
        expect(receivedBody).toEqual({ name: 'Renamed' });
        expect(receivedId).toBe(model.id);
        expect(queryClient.getQueryState(modelKey)?.isInvalidated).toBe(true);
        expect(queryClient.getQueryState(listKey)?.isInvalidated).toBe(true);
    });

    it('disables Save for blank or unchanged names', async () => {
        const user = userEvent.setup();
        render(<RenameModelDialog model={model} close={rs.fn()} />);

        const name = screen.getByRole('textbox', { name: /Model name/ });
        const save = screen.getByRole('button', { name: 'Save' });
        await user.clear(name);
        expect(save).toBeDisabled();
        await user.type(name, '   ');
        expect(save).toBeDisabled();
        await user.type(name, model.name);
        expect(save).toBeDisabled();
    });

    it('keeps the dialog open and shows an error when renaming fails', async () => {
        const user = userEvent.setup();
        const close = rs.fn();
        server.use(http.patch('/api/models/{model_id}', () => HttpResponse.json({}, { status: 500 })));
        render(<RenameModelDialog model={model} close={close} />);

        await user.type(screen.getByRole('textbox', { name: /Model name/ }), '-renamed');
        await user.click(screen.getByRole('button', { name: 'Save' }));

        expect(await screen.findByText('The model could not be renamed. Try again.')).toBeInTheDocument();
        expect(close).not.toHaveBeenCalled();
        expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled();
    });
});
