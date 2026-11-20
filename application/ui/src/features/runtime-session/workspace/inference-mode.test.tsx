import { rs } from '@rstest/core';
import { fireEvent, screen } from '@testing-library/react';

import { render } from '../../../test-utils/render';
import { InferenceMode } from './inference-mode';

const send = rs.fn(async () => ({ ok: true }));

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: () => null,
    useRuntimeSession: () => ({
        state: {
            environment: { name: 'Bench', cameras: [] },
            model: { model_id: 'm', name: 'ACT', backend: 'torch', device: 'cpu', tasks: ['pick', 'place'] },
        },
        statuses: {},
        send,
    }),
}));

describe('InferenceMode', () => {
    it('runs a multi-task model on one of its tasks, without free text', async () => {
        render(<InferenceMode />);

        expect(screen.queryByRole('textbox', { name: 'Task' })).not.toBeInTheDocument();
        fireEvent.click(await screen.findByRole('button', { name: 'Play' }));

        expect(send).toHaveBeenCalledWith(
            expect.objectContaining({ control: expect.objectContaining({ type: 'model', task: 'pick' }) })
        );
    });
});
