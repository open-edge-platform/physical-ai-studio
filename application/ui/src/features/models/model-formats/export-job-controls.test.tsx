import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedModelExportJob } from '../../../test-utils/mocks/mock-model-export-job';
import { render } from '../../../test-utils/render';
import { ExportJobControls } from './export-job-controls';

describe('ExportJobControls', () => {
    it('queues an export for the selected missing backend', async () => {
        const submitted = rs.fn();
        server.use(
            http.get('/api/jobs', () => HttpResponse.json([])),
            http.post('/api/models/{model_id}/exports/{backend}', ({ params }) => {
                submitted(params);
                return HttpResponse.json(getMockedModelExportJob(), { status: 202 });
            })
        );
        const user = userEvent.setup();
        render(<ExportJobControls modelId='model-1' action={{ backend: 'torch', operation: 'export' }} />);
        await waitFor(() => expect(screen.getByRole('button', { name: 'Export model' })).toBeEnabled());
        await user.click(screen.getByRole('button', { name: 'Export model' }));
        await waitFor(() => expect(submitted).toHaveBeenCalledWith({ model_id: 'model-1', backend: 'torch' }));
    });

    it('blocks another export and allows canceling the active job', async () => {
        const canceled = rs.fn();
        server.use(
            http.get('/api/jobs', () => HttpResponse.json([getMockedModelExportJob({ status: 'running' })])),
            http.post('/api/jobs/{job_id}:interrupt', () => {
                canceled();
                return new HttpResponse(null, { status: 204 });
            })
        );
        const user = userEvent.setup();
        render(<ExportJobControls modelId='model-1' action={{ backend: 'onnx', operation: 'export' }} />);
        await user.click(await screen.findByRole('button', { name: 'Cancel job' }));
        expect(screen.getByRole('button', { name: 'Export model' })).toBeDisabled();
        await waitFor(() => expect(canceled).toHaveBeenCalled());
    });
});
