import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';

import { render } from '../../../test-utils/render';
import { PolicySelection } from './policy-selection';

const policies = [
    { id: 'act', name: 'ACT', description: 'ACT policy', min_vram: 2 * 1024 ** 3 },
    { id: 'pi05', name: 'Pi0.5', description: 'Pi0.5 policy', min_vram: 40 * 1024 ** 3 },
    { id: 'rldx1', name: 'RLDX-1', description: 'RLDX-1 policy', min_vram: 40 * 1024 ** 3 },
];

const renderWithVram = (gib: number) =>
    render(
        <PolicySelection
            selectedPolicy='pi05'
            availablePolicies={policies}
            isLoadingPolicies={false}
            onSelectionChange={rs.fn()}
            trainingDevice={{ type: 'cuda', name: 'A100', index: 0, memory: gib * 1024 ** 3 }}
        />
    );

describe('PolicySelection VRAM warning', () => {
    it('does not warn when reported VRAM and the policy card both display 40 GB', () => {
        renderWithVram(39.4);
        expect(screen.getAllByText('≥ 40 GB VRAM')).toHaveLength(2);
        expect(screen.queryByText(/Training may fail/)).not.toBeInTheDocument();
    });

    it('warns when displayed VRAM is below the policy requirement', () => {
        renderWithVram(38.4);
        expect(screen.getByText(/Training may fail/)).toBeInTheDocument();
    });

    it('does not offer policies missing from the backend response', () => {
        render(
            <PolicySelection
                selectedPolicy='act'
                availablePolicies={policies.slice(0, 1)}
                isLoadingPolicies={false}
                onSelectionChange={rs.fn()}
                trainingDevice={null}
            />
        );
        expect(screen.getByLabelText('Select ACT policy')).toBeInTheDocument();
        expect(screen.queryByLabelText('Select Pi0.5 policy')).not.toBeInTheDocument();
    });

    it('shows a new library policy without a frontend registry entry', () => {
        render(
            <PolicySelection
                selectedPolicy='cosmos3'
                availablePolicies={[{ id: 'cosmos3', name: 'Cosmos3', description: 'World model', min_vram: null }]}
                isLoadingPolicies={false}
                onSelectionChange={rs.fn()}
                trainingDevice={null}
            />
        );
        expect(screen.getByLabelText('Select Cosmos3 policy')).toBeInTheDocument();
        expect(screen.getByText('World model')).toBeInTheDocument();
    });
});
