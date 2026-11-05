import { rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { render } from '../../../test-utils/render';
import { StartInferenceDialog } from './start-inference-dialog';

const CurrentLocation = () => {
    const { pathname, search } = useLocation();
    return <output aria-label='Current location'>{pathname + search}</output>;
};

describe('StartInferenceDialog model selection', () => {
    it.each([
        getMockedModel(),
        getMockedModel({
            id: 'optimized-model',
            name: 'pick-and-place (INT8)',
            train_job_id: null,
            optimization: { source_model_id: 'model-1', mode: 'int8_sym' },
        }),
    ])('runs the selected row using its own model id ($id) and the ordinary OpenVINO backend', async (model) => {
        server.use(
            http.get('/api/policies/backends', () => HttpResponse.json({ act: ['openvino', 'torch'] })),
            http.get('/api/system/devices/inference', () =>
                HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
            )
        );
        const close = rs.fn();
        const user = userEvent.setup();
        render(
            <>
                <StartInferenceDialog model={model} close={close} />
                <CurrentLocation />
            </>,
            { path: '*', route: '/projects/project-1/models' }
        );
        await waitFor(() => expect(screen.getByRole('button', { name: 'Start' })).toBeEnabled());
        await user.click(screen.getByRole('button', { name: 'Start' }));

        expect(screen.getByLabelText('Current location')).toHaveTextContent(
            `/projects/project-1/models/${model.id}/inference/openvino?device=CPU`
        );
        expect(close).toHaveBeenCalledTimes(1);
    });
});
