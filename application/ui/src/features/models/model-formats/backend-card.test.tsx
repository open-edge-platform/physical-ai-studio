import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { render } from '../../../test-utils/render';
import { BackendCard } from './backend-card';

it('selects the INT8 package for downloads while retaining the existing backend card', async () => {
    server.use(http.get('/api/jobs', () => HttpResponse.json([])));
    const detail = getMockedModelDetailResponse();
    detail.exports[0].variants = [
        getMockedExportVariant(),
        getMockedExportVariant({ id: 'int8', weight_compression: 'int8_sym' }),
    ];
    const user = userEvent.setup();
    render(<BackendCard model={detail.model} modelDetail={detail} backendType='openvino' />);
    const download = screen.getByRole('button', { name: 'Download OpenVINO export' });
    expect(download).toHaveAttribute('href', expect.stringContaining('variant=original'));
    await user.click(screen.getByRole('button', { name: /Variant/ }));
    await user.click(await screen.findByRole('option', { name: 'INT8' }));
    expect(download).toHaveAttribute('href', expect.stringContaining('variant=int8'));
    expect(screen.getByText('INT8 (weights)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
});
