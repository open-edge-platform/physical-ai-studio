import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { render } from '../../../test-utils/render';
import { BackendSelection } from './backend-selection';

it('offers existing exports when the training policy is absent from the catalog', async () => {
    server.use(http.get('/api/policies/backends', () => HttpResponse.json({})));
    render(
        <BackendSelection
            model={getMockedModel({ policy: 'removed_policy', available_backends: ['openvino', 'onnx'] })}
            inferenceDevice={undefined}
            inferenceDevices={[]}
            setInferenceDevice={rs.fn()}
        />
    );
    expect(await screen.findByText('OpenVINO')).toBeInTheDocument();
    expect(screen.getByRole('radio')).toBeEnabled();
});
