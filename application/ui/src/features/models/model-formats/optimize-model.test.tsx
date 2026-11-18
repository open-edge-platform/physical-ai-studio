import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaOptimizeModelJob } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedOptimizeModelJob } from '../../../test-utils/mocks/mock-optimize-model-job';
import { render } from '../../../test-utils/render';
import { OptimizeModel } from './optimize-model';

it('queues an INT8 job, shows its status, and cancels it', async () => {
    let jobs: SchemaOptimizeModelJob[] = [];
    server.use(
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.post('/api/models/{model_id}/optimize', ({ params }) => {
            expect(params.model_id).toBe('model-1');
            jobs = [getMockedOptimizeModelJob()];
            return HttpResponse.json(jobs[0], { status: 202 });
        }),
        http.post('/api/jobs/{job_id}:interrupt', () => {
            jobs = [getMockedOptimizeModelJob({ status: 'canceled' })];
            return HttpResponse.json(jobs[0]);
        })
    );
    const user = userEvent.setup();
    render(<OptimizeModel modelId='model-1' hasInt8={false} />);
    const button = screen.getByRole('button', { name: 'Optimize to INT8' });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);
    expect(await screen.findByRole('status')).toHaveTextContent('Optimization pending');
    expect(button).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Cancel optimization' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Optimization canceled'));
    await waitFor(() => expect(button).toBeEnabled());
});

it('does not queue a duplicate when an INT8 export already exists', async () => {
    server.use(http.get('/api/jobs', () => HttpResponse.json([])));
    render(<OptimizeModel modelId='model-1' hasInt8 />);
    expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
});

it('blocks submission while job status is unavailable and supports retry', async () => {
    server.use(http.get('/api/jobs', () => HttpResponse.error()));
    const user = userEvent.setup();
    render(<OptimizeModel modelId='model-1' hasInt8={false} />);
    const retry = await screen.findByRole('button', { name: 'Retry optimization status' });
    expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
    server.use(http.get('/api/jobs', () => HttpResponse.json([])));
    await user.click(retry);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled());
});
