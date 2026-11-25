import { rs } from '@rstest/core';
import { fireEvent, screen } from '@testing-library/react';

import { render } from '../../../test-utils/render';
import { PartsPanel } from './parts-panel';

const statuses = {
    environment: { type: 'status', component: 'environment', state: 'running' },
    'environment/follower': { type: 'status', component: 'environment/follower', state: 'running' },
    'environment/leader': { type: 'status', component: 'environment/leader', state: 'failed', error: 'Unplugged' },
};

rs.mock('../runtime-session-provider', () => ({
    failureOf: () => null,
    useRuntimeSession: () => ({ state: { environment: { name: 'Bench', cameras: [] } }, statuses }),
}));
rs.mock('./part-controls', () => ({
    usePartControls: () => ({ pendingPart: null, act: () => undefined }),
}));

describe('PartsPanel', () => {
    it('groups the devices under the environment, and expands to show which failed and why', async () => {
        render(<PartsPanel />);

        expect(await screen.findByText(/1 device failed/)).toBeInTheDocument();
        expect(screen.queryByText('Leader')).not.toBeInTheDocument();
        expect(screen.queryByText('Unplugged')).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Show devices' }));

        expect(screen.getByText('Follower')).toBeInTheDocument();
        expect(screen.getByText('Leader')).toBeInTheDocument();
        expect(screen.getByText('Unplugged')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Hide devices' })).toHaveAttribute('aria-expanded', 'true');
    });
});
