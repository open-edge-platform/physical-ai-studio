import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedModelExportJob } from '../../../test-utils/mocks/mock-model-export-job';
import { render } from '../../../test-utils/render';
import { ExportJobControls } from './export-job-controls';

describe('ExportJobControls', () => {
    it('announces the request without a progress bar', () => {
        render(<ExportJobControls label='PyTorch' isSubmitting isRetryDisabled />);
        expect(screen.getByRole('status')).toHaveTextContent('Requesting export');
        expect(screen.getByRole('status', { name: 'PyTorch export status' })).toHaveTextContent('Requesting export');
        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
        expect(screen.getByText('You can keep working while this export finishes.')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Cancel export' })).not.toBeInTheDocument();
    });

    it('announces backend-specific activity and allows canceling the job', async () => {
        const canceled = rs.fn();
        server.use(
            http.post('/api/jobs/{job_id}:interrupt', ({ params }) => {
                canceled(params.job_id);
                return new HttpResponse(null, { status: 204 });
            })
        );
        const user = userEvent.setup();
        render(
            <ExportJobControls
                label='PyTorch'
                job={getMockedModelExportJob({ status: 'running', progress: 35 })}
                isSubmitting={false}
                isRetryDisabled
            />
        );
        expect(screen.getByRole('status', { name: 'PyTorch export status' })).toHaveTextContent('Exporting variant');
        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Cancel export' }));
        await waitFor(() => expect(canceled).toHaveBeenCalledWith('export-job-1'));
        expect(await screen.findByText('Canceling export…')).toBeInTheDocument();
    });

    it('allows retrying a failed export without showing progress', async () => {
        const retry = rs.fn();
        render(
            <ExportJobControls
                label='PyTorch'
                job={getMockedModelExportJob({ status: 'failed', message: 'Not enough memory' })}
                isSubmitting={false}
                isRetryDisabled={false}
                onRetry={retry}
            />
        );
        expect(screen.getByRole('status')).toHaveTextContent('Not enough memory');
        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
        await userEvent.setup().click(screen.getByRole('button', { name: 'Retry export' }));
        expect(retry).toHaveBeenCalledOnce();
    });
});
