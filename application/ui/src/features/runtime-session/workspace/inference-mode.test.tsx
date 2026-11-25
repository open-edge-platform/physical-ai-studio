import { rs } from '@rstest/core';
import { fireEvent, screen } from '@testing-library/react';

import { render } from '../../../test-utils/render';
import { InferenceMode } from './inference-mode';

const send = rs.fn(async () => ({ ok: true }));
const ACT = { model_id: 'm', name: 'ACT', backend: 'torch', device: 'cpu', tasks: ['pick', 'place'] };
let model: Record<string, unknown> = ACT;

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: () => null,
    useRuntimeSession: () => ({
        state: { environment: { name: 'Bench', cameras: [] }, model },
        statuses: {},
        send,
    }),
}));

const chooseRtc = async () => {
    fireEvent.click(await screen.findByRole('button', { name: /Execution/ }));
    fireEvent.click(await screen.findByRole('option', { name: 'Real-time chunking (RTC)' }));
};

describe('InferenceMode', () => {
    beforeEach(() => {
        model = ACT;
        send.mockClear();
    });

    it('runs an RTC export with its horizon and guidance, and measures the latency unless told', async () => {
        model = { model_id: 'p', name: 'pi05', backend: 'openvino', device: 'GPU', tasks: [], rtc: true };
        render(<InferenceMode />);

        await chooseRtc();
        expect(screen.getByRole('slider', { name: 'Guided actions (horizon)' })).toBeInTheDocument();
        expect(screen.getByRole('slider', { name: 'Expected inference latency' })).toBeInTheDocument();
        expect(screen.queryByRole('slider', { name: 'Request the next chunk at' })).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Play' }));

        expect(send).toHaveBeenCalledWith(
            expect.objectContaining({
                control: expect.objectContaining({
                    execution: expect.objectContaining({
                        mode: 'rtc',
                        execution_horizon: 25,
                        max_guidance_weight: 5,
                        initial_latency_s: null,
                    }),
                }),
            })
        );
    });

    it('will not run RTC on an export without RTC inputs, and says why', async () => {
        render(<InferenceMode />);

        await chooseRtc();

        expect(screen.getByText(/exported without RTC inputs/)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
    });

    it('runs a multi-task model on one of its tasks, without free text', async () => {
        render(<InferenceMode />);

        expect(screen.queryByRole('textbox', { name: 'Task' })).not.toBeInTheDocument();
        fireEvent.click(await screen.findByRole('button', { name: 'Play' }));

        expect(send).toHaveBeenCalledWith(
            expect.objectContaining({ control: expect.objectContaining({ type: 'model', task: 'pick' }) })
        );
    });
});
