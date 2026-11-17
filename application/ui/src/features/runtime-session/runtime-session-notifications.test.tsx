import { toast } from '@geti-ui/ui';
import { afterEach, describe, expect, it, rs } from '@rstest/core';
import { act } from '@testing-library/react';
import useWebSocket from 'react-use-websocket';

import { render } from '../../test-utils/render';
import { ERROR_TOAST, RuntimeSessionProvider } from './runtime-session-provider';

rs.mock('react-use-websocket', () => ({
    default: rs.fn(() => ({ sendJsonMessage: rs.fn() })),
}));

afterEach(() => {
    rs.restoreAllMocks();
    rs.clearAllMocks();
});

describe('recording notifications', () => {
    it('shows why an episode was discarded even though the dataset worker is still running', () => {
        const notify = rs.spyOn(toast, 'negative').mockReturnValue(rs.fn());
        render(
            <RuntimeSessionProvider project_id='project' environment={null}>
                {null}
            </RuntimeSessionProvider>
        );
        const onMessage = rs.mocked(useWebSocket).mock.calls.at(-1)?.[1]?.onMessage;
        expect(onMessage).toBeDefined();
        const error = 'Episode discarded because camera "front" stopped sending frames.';

        act(() => {
            onMessage?.(
                new MessageEvent('message', {
                    data: JSON.stringify({ type: 'status', component: 'dataset', state: 'running', error }),
                })
            );
        });

        expect(notify).toHaveBeenCalledWith(error, ERROR_TOAST);
    });

    it('does not notify for ordinary dataset status updates', () => {
        const notify = rs.spyOn(toast, 'negative').mockReturnValue(rs.fn());
        render(
            <RuntimeSessionProvider project_id='project' environment={null}>
                {null}
            </RuntimeSessionProvider>
        );
        const onMessage = rs.mocked(useWebSocket).mock.calls.at(-1)?.[1]?.onMessage;

        act(() => {
            onMessage?.(
                new MessageEvent('message', {
                    data: JSON.stringify({ type: 'status', component: 'dataset', state: 'running' }),
                })
            );
        });

        expect(notify).not.toHaveBeenCalled();
    });
});
