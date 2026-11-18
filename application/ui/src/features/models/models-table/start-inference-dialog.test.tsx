import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { render } from '../../../test-utils/render';
import { StartInferenceDialog } from './start-inference-dialog';

const Location = () => {
    const { pathname, search } = useLocation();
    return <output aria-label='Current location'>{pathname + search}</output>;
};

it.each([false, true])('starts the selected export and resets it when changing backends: %s', async (switchBackend) => {
    const detail = getMockedModelDetailResponse();
    detail.model = { ...detail.model, available_backends: ['openvino', 'torch'] };
    detail.exports[0].variants = [getMockedExportVariant(), getMockedExportVariant({ id: 'int8' })];
    detail.exports.push({ type: 'torch', size_bytes: 4096, file_count: 3, variants: [getMockedExportVariant()] });
    server.use(
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/policies/backends', () => HttpResponse.json({ act: ['openvino', 'torch'] })),
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([
                { type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' },
                { type: 'cpu', name: 'CPU', backend: 'torch', device: 'cpu' },
            ])
        )
    );
    const user = userEvent.setup();
    render(
        <>
            <StartInferenceDialog model={detail.model} close={() => {}} />
            <Location />
        </>,
        { path: '*', route: '/projects/project-1/models' }
    );
    await user.click(await screen.findByRole('button', { name: /Variant/ }));
    await user.click(await screen.findByRole('option', { name: 'INT8' }));
    if (switchBackend) await user.click(screen.getByRole('radio', { name: /PyTorch/ }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start' }));
    expect(screen.getByLabelText('Current location')).toHaveTextContent(
        switchBackend ? '/inference/torch?device=cpu&variant=original' : '/inference/openvino?device=CPU&variant=int8'
    );
});
