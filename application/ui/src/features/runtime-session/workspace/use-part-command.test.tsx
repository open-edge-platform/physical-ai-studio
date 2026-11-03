import { rs } from '@rstest/core';
import { act } from '@testing-library/react';

import { renderHook } from '../../../test-utils/render';
import { usePartCommand } from './use-part-command';

const send = rs.fn();
rs.mock('../runtime-session-provider', () => ({
    useRuntimeSession: () => ({ send }),
}));

const draft = { name: 'Bench', robots: [], cameras: [] };
const incompatible = {
    type: 'ack',
    request_id: '1',
    ok: false,
    error: {
        message: 'This would unload dataset',
        error_code: 'incompatible_parts',
        parts: [{ part: 'dataset', name: 'Blocks', reasons: ["camera 'wrist': no such camera"] }],
    },
};

describe('usePartCommand', () => {
    beforeEach(() => send.mockReset());

    it('asks before unloading parts that no longer fit, then loads again with unload_incompatible', async () => {
        send.mockResolvedValueOnce(incompatible).mockResolvedValueOnce({ type: 'ack', request_id: '2', ok: true });
        const { result } = renderHook(() => usePartCommand());

        await act(async () => {
            await result.current.run({ command: 'load_environment', environment: draft, unload_incompatible: false });
        });
        expect(result.current.confirmation?.parts[0].name).toBe('Blocks');
        expect(send).toHaveBeenCalledTimes(1);

        await act(async () => {
            result.current.confirmation?.confirm();
        });
        expect(send).toHaveBeenLastCalledWith({
            command: 'load_environment',
            environment: draft,
            unload_incompatible: true,
        });
        expect(result.current.confirmation).toBeNull();
    });

    it('cancelling leaves things as they are', async () => {
        send.mockResolvedValueOnce(incompatible);
        const { result } = renderHook(() => usePartCommand());

        await act(async () => {
            await result.current.run({ command: 'load_environment', environment: draft, unload_incompatible: false });
        });
        act(() => result.current.confirmation?.cancel());

        expect(result.current.confirmation).toBeNull();
        expect(send).toHaveBeenCalledTimes(1);
    });
});
