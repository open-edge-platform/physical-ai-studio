import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { render } from '../../../test-utils/render';
import { BackendCard } from './backend-card';

const Location = () => {
    const { pathname, search } = useLocation();
    return <output aria-label='Current location'>{pathname + search}</output>;
};

it('downloads and runs the fixed package without a variant picker', async () => {
    server.use(
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
        )
    );
    const user = userEvent.setup();
    render(
        <>
            <BackendCard
                model={getMockedModel()}
                variant={getMockedExportVariant({ id: 'int8', weight_compression: 'int8_sym' })}
                backendType='openvino'
                onCreate={() => {}}
            />
            <Location />
        </>,
        { path: '*', route: '/projects/project-1/models' }
    );
    expect(screen.getByRole('region', { name: 'OpenVINO · INT8' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Variant/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Actions for OpenVINO INT8' }));
    expect(await screen.findByRole('menuitem', { name: 'Download' })).toHaveAttribute(
        'href',
        expect.stringContaining('variant=int8')
    );
    await user.keyboard('{Escape}');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Run' }));
    expect(screen.getByLabelText('Current location')).toHaveTextContent('/inference/openvino?device=CPU&variant=int8');
});

it('does not offer Studio inference for ONNX packages', async () => {
    render(
        <BackendCard
            model={getMockedModel()}
            variant={getMockedExportVariant()}
            backendType='onnx'
            onCreate={() => {}}
        />
    );
    expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Optimize' })).not.toBeInTheDocument();
});
